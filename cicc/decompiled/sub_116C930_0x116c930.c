// Function: sub_116C930
// Address: 0x116c930
//
__int64 __fastcall sub_116C930(__int64 a1, _BYTE *a2, char a3, unsigned int a4)
{
  _BYTE *v4; // r15
  __int64 v8; // rdi
  _BYTE *v10; // rdi
  char v11; // [rsp+Fh] [rbp-F1h]
  _QWORD v12[2]; // [rsp+10h] [rbp-F0h] BYREF
  _BYTE *v13; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v14; // [rsp+28h] [rbp-D8h]
  _BYTE v15[64]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v16; // [rsp+70h] [rbp-90h] BYREF
  char *v17; // [rsp+78h] [rbp-88h]
  __int64 v18; // [rsp+80h] [rbp-80h]
  int v19; // [rsp+88h] [rbp-78h]
  char v20; // [rsp+8Ch] [rbp-74h]
  char v21; // [rsp+90h] [rbp-70h] BYREF

  v4 = a2;
  if ( (unsigned __int8)(*a2 - 12) <= 1u )
    return (__int64)v4;
  if ( (unsigned __int8)(*a2 - 9) <= 2u )
  {
    v16 = 0;
    v17 = &v21;
    v13 = v15;
    v18 = 8;
    v19 = 0;
    v20 = 1;
    v14 = 0x800000000LL;
    v12[0] = &v16;
    v12[1] = &v13;
    v11 = sub_AA8FD0(v12, (__int64)a2);
    if ( v11 )
    {
      while ( 1 )
      {
        v10 = v13;
        if ( !(_DWORD)v14 )
          break;
        a2 = *(_BYTE **)&v13[8 * (unsigned int)v14 - 8];
        LODWORD(v14) = v14 - 1;
        if ( !(unsigned __int8)sub_AA8FD0(v12, (__int64)a2) )
          goto LABEL_7;
      }
    }
    else
    {
LABEL_7:
      v11 = 0;
      v10 = v13;
    }
    if ( v10 != v15 )
      _libc_free(v10, a2);
    if ( !v20 )
      _libc_free(v17, a2);
    if ( v11 )
      return (__int64)v4;
  }
  v8 = *((_QWORD *)v4 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
    v8 = **(_QWORD **)(v8 + 16);
  if ( sub_BCAC40(v8, 1) )
    return (__int64)v4;
  else
    return sub_1169C30(a1, v4, a3, a4);
}

// Function: sub_1003090
// Address: 0x1003090
//
__int64 __fastcall sub_1003090(__int64 a1, unsigned __int8 *a2)
{
  unsigned int v2; // r12d
  int v4; // eax
  _BYTE *v5; // rdi
  _QWORD v6[2]; // [rsp+0h] [rbp-E0h] BYREF
  _BYTE *v7; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v8; // [rsp+18h] [rbp-C8h]
  _BYTE v9[64]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v10; // [rsp+60h] [rbp-80h] BYREF
  char *v11; // [rsp+68h] [rbp-78h]
  __int64 v12; // [rsp+70h] [rbp-70h]
  int v13; // [rsp+78h] [rbp-68h]
  char v14; // [rsp+7Ch] [rbp-64h]
  char v15; // [rsp+80h] [rbp-60h] BYREF

  v2 = *(unsigned __int8 *)(a1 + 65);
  if ( (_BYTE)v2 )
  {
    v4 = *a2;
    if ( (unsigned int)(v4 - 12) > 1 )
    {
      v2 = 0;
      if ( (unsigned int)(v4 - 9) <= 2 )
      {
        v10 = 0;
        v11 = &v15;
        v14 = 1;
        v12 = 8;
        v13 = 0;
        v7 = v9;
        v8 = 0x800000000LL;
        v6[0] = &v10;
        v6[1] = &v7;
        v2 = sub_AA8FD0(v6, (__int64)a2);
        if ( (_BYTE)v2 )
        {
          while ( 1 )
          {
            v5 = v7;
            if ( !(_DWORD)v8 )
              break;
            a2 = *(unsigned __int8 **)&v7[8 * (unsigned int)v8 - 8];
            LODWORD(v8) = v8 - 1;
            if ( !(unsigned __int8)sub_AA8FD0(v6, (__int64)a2) )
              goto LABEL_6;
          }
        }
        else
        {
LABEL_6:
          v5 = v7;
          v2 = 0;
        }
        if ( v5 != v9 )
          _libc_free(v5, a2);
        if ( !v14 )
          _libc_free(v11, a2);
      }
    }
  }
  return v2;
}

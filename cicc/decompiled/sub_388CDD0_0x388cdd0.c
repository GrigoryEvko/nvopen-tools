// Function: sub_388CDD0
// Address: 0x388cdd0
//
__int64 __fastcall sub_388CDD0(__int64 a1, _BYTE *a2)
{
  unsigned int v2; // r13d
  __int64 v4; // r15
  int v7; // eax
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rsi
  const char *v11; // rdi
  int v12; // eax
  _QWORD *v13; // rdi
  __int64 v14; // rsi
  char v15; // al
  unsigned __int64 v16; // [rsp+8h] [rbp-78h]
  _QWORD v17[2]; // [rsp+10h] [rbp-70h] BYREF
  char v18; // [rsp+20h] [rbp-60h]
  char v19; // [rsp+21h] [rbp-5Fh]
  const char *v20; // [rsp+30h] [rbp-50h] BYREF
  __int64 v21; // [rsp+38h] [rbp-48h]
  _WORD v22[32]; // [rsp+40h] [rbp-40h] BYREF

  v2 = 0;
  *a2 = 1;
  if ( *(_DWORD *)(a1 + 64) == 74 )
  {
    v4 = a1 + 8;
    v7 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v7;
    if ( v7 != 12 )
    {
      v8 = *(_QWORD *)(a1 + 56);
      v22[0] = 259;
      v20 = "Expected '(' in syncscope";
      return (unsigned int)sub_38814C0(v4, v8, (__int64)&v20);
    }
    v20 = (const char *)v22;
    *(_DWORD *)(a1 + 64) = sub_3887100(v4);
    v9 = *(_QWORD *)(a1 + 56);
    v21 = 0;
    LOBYTE(v22[0]) = 0;
    v16 = v9;
    v2 = sub_388B0A0(a1, (unsigned __int64 *)&v20);
    if ( (_BYTE)v2 )
    {
      v19 = 1;
      v18 = 3;
      v17[0] = "Expected synchronization scope name";
      v2 = sub_38814C0(v4, v16, (__int64)v17);
    }
    else
    {
      v10 = *(_QWORD *)(a1 + 56);
      if ( *(_DWORD *)(a1 + 64) == 13 )
      {
        v12 = sub_3887100(v4);
        v13 = *(_QWORD **)a1;
        v14 = (__int64)v20;
        *(_DWORD *)(a1 + 64) = v12;
        v15 = sub_16032D0(v13, v14, v21);
        v11 = v20;
        *a2 = v15;
        if ( v11 == (const char *)v22 )
          return v2;
        goto LABEL_9;
      }
      v19 = 1;
      v17[0] = "Expected ')' in syncscope";
      v18 = 3;
      v2 = sub_38814C0(v4, v10, (__int64)v17);
    }
    v11 = v20;
    if ( v20 == (const char *)v22 )
      return v2;
LABEL_9:
    j_j___libc_free_0((unsigned __int64)v11);
  }
  return v2;
}

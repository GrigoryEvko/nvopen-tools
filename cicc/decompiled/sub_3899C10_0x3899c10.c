// Function: sub_3899C10
// Address: 0x3899c10
//
__int64 __fastcall sub_3899C10(__int64 a1, _BYTE *a2, size_t a3, __int64 *a4)
{
  __int64 result; // rax
  __int64 v5; // r15
  int v9; // eax
  __int64 v10; // r8
  unsigned __int64 v11; // rax
  const char *v12; // rax
  unsigned __int64 v13; // rsi
  _QWORD *v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rdi
  int v17; // eax
  _QWORD *v18; // rdi
  size_t v19; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v20; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v21; // [rsp+28h] [rbp-48h]
  _QWORD v22[8]; // [rsp+30h] [rbp-40h] BYREF

  result = 0;
  *a4 = 0;
  v5 = *(_QWORD *)(a1 + 56);
  if ( *(_DWORD *)(a1 + 64) != 201 )
    return result;
  v9 = sub_3887100(a1 + 8);
  v10 = a1 + 8;
  *(_DWORD *)(a1 + 64) = v9;
  if ( v9 != 12 )
  {
    v11 = a3;
    if ( !a3 )
    {
      BYTE1(v22[0]) = 1;
      v12 = "comdat cannot be unnamed";
LABEL_5:
      v13 = *(_QWORD *)(a1 + 56);
      v20 = v12;
      LOBYTE(v22[0]) = 3;
      return sub_38814C0(v10, v13, (__int64)&v20);
    }
    if ( !a2 )
    {
      v21 = 0;
      v20 = v22;
      LOBYTE(v22[0]) = 0;
      goto LABEL_11;
    }
    v19 = a3;
    v20 = v22;
    if ( a3 > 0xF )
    {
      v20 = (_QWORD *)sub_22409D0((__int64)&v20, &v19, 0);
      v18 = v20;
      v22[0] = v19;
    }
    else
    {
      if ( a3 == 1 )
      {
        LOBYTE(v22[0]) = *a2;
        v14 = v22;
LABEL_10:
        v21 = v11;
        *((_BYTE *)v14 + v11) = 0;
LABEL_11:
        v15 = sub_3899950((__int64 *)a1, (__int64)&v20, v5);
        v16 = v20;
        *a4 = v15;
        if ( v16 != v22 )
          j_j___libc_free_0((unsigned __int64)v16);
        return 0;
      }
      v18 = v22;
    }
    memcpy(v18, a2, a3);
    v11 = v19;
    v14 = v20;
    goto LABEL_10;
  }
  v17 = sub_3887100(v10);
  v10 = a1 + 8;
  *(_DWORD *)(a1 + 64) = v17;
  if ( v17 != 374 )
  {
    BYTE1(v22[0]) = 1;
    v12 = "expected comdat variable";
    goto LABEL_5;
  }
  *a4 = sub_3899950((__int64 *)a1, a1 + 72, *(_QWORD *)(a1 + 56));
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  return sub_388AF10(a1, 13, "expected ')' after comdat var");
}

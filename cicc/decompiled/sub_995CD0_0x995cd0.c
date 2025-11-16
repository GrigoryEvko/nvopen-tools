// Function: sub_995CD0
// Address: 0x995cd0
//
__int64 __fastcall sub_995CD0(unsigned __int8 *a1)
{
  unsigned __int8 v1; // al
  _BYTE *v2; // rsi
  unsigned int v3; // eax
  unsigned __int64 v4; // rdx
  _QWORD *v5; // rdx
  __int64 v6; // rdi
  __int64 result; // rax
  _BYTE *v8; // rax
  char v9; // al
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+8h] [rbp-48h]
  __int64 v17; // [rsp+18h] [rbp-38h] BYREF
  _QWORD *v18; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-28h]
  _QWORD *v20; // [rsp+30h] [rbp-20h] BYREF
  __int64 *v21; // [rsp+38h] [rbp-18h]

  v21 = &v17;
  v1 = *a1;
  v20 = 0;
  if ( v1 == 59 )
  {
    v9 = sub_995B10(&v20, *((_QWORD *)a1 - 8));
    v10 = *((_QWORD *)a1 - 4);
    if ( v9 && v10 )
    {
      *v21 = v10;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(&v20, v10) || (v11 = *((_QWORD *)a1 - 8)) == 0 )
      {
        v1 = *a1;
        goto LABEL_2;
      }
      *v21 = v11;
    }
    return v17;
  }
LABEL_2:
  v2 = a1 + 24;
  if ( v1 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a1 + 1) + 8LL) - 17 > 1 )
      return 0;
    if ( v1 > 0x15u )
      return 0;
    v8 = (_BYTE *)sub_AD7630(a1, 0);
    if ( !v8 )
      return 0;
    v2 = v8 + 24;
    if ( *v8 != 17 )
      return 0;
  }
  v3 = *((_DWORD *)v2 + 2);
  LODWORD(v21) = v3;
  if ( v3 <= 0x40 )
  {
    v4 = *(_QWORD *)v2;
LABEL_5:
    v5 = (_QWORD *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v3) & ~v4);
    if ( !v3 )
      v5 = 0;
    v20 = v5;
    goto LABEL_8;
  }
  sub_C43780(&v20, v2);
  v3 = (unsigned int)v21;
  if ( (unsigned int)v21 <= 0x40 )
  {
    v4 = (unsigned __int64)v20;
    goto LABEL_5;
  }
  sub_C43D10(&v20, v2, v12, v13, v14);
  v3 = (unsigned int)v21;
  v5 = v20;
LABEL_8:
  v6 = *((_QWORD *)a1 + 1);
  v19 = v3;
  v18 = v5;
  LODWORD(v21) = 0;
  result = sub_AD8D80(v6, &v18);
  if ( v19 > 0x40 && v18 )
  {
    v15 = result;
    j_j___libc_free_0_0(v18);
    result = v15;
  }
  if ( (unsigned int)v21 > 0x40 )
  {
    if ( v20 )
    {
      v16 = result;
      j_j___libc_free_0_0(v20);
      return v16;
    }
  }
  return result;
}

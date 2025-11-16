// Function: sub_1C42AC0
// Address: 0x1c42ac0
//
__int64 __fastcall sub_1C42AC0(__int64 *a1, __int64 a2, int a3, int a4, __int64 a5, __int64 a6, __int64 a7, __int64 a8)
{
  char *v9; // r14
  char *v10; // rbx
  __int64 v11; // rdi
  __int64 v12; // rax
  char *v13; // rdi
  __int64 result; // rax
  __int64 v15; // r11
  char *v16; // rsi
  bool v21; // [rsp+28h] [rbp-48h]
  __int64 v22; // [rsp+28h] [rbp-48h]
  __int64 v23[7]; // [rsp+38h] [rbp-38h] BYREF

  v9 = (char *)a1[2];
  v10 = (char *)a1[1];
  v11 = (v9 - v10) >> 5;
  v12 = (v9 - v10) >> 3;
  if ( v11 > 0 )
  {
    v13 = &v10[32 * v11];
    while ( a2 != **(_QWORD **)v10 )
    {
      if ( a2 == **((_QWORD **)v10 + 1) )
      {
        v10 += 8;
        v21 = v9 == v10;
        goto LABEL_9;
      }
      if ( a2 == **((_QWORD **)v10 + 2) )
      {
        v10 += 16;
        v21 = v9 == v10;
        goto LABEL_9;
      }
      if ( a2 == **((_QWORD **)v10 + 3) )
      {
        v10 += 24;
        v21 = v9 == v10;
        goto LABEL_9;
      }
      v10 += 32;
      if ( v13 == v10 )
      {
        v12 = (v9 - v10) >> 3;
        goto LABEL_20;
      }
    }
    goto LABEL_8;
  }
LABEL_20:
  if ( v12 == 2 )
  {
LABEL_26:
    if ( a2 == **(_QWORD **)v10 )
      goto LABEL_8;
    v10 += 8;
    goto LABEL_28;
  }
  if ( v12 == 3 )
  {
    if ( a2 == **(_QWORD **)v10 )
      goto LABEL_8;
    v10 += 8;
    goto LABEL_26;
  }
  if ( v12 != 1 )
  {
LABEL_23:
    v21 = 1;
    v10 = v9;
    goto LABEL_9;
  }
LABEL_28:
  if ( a2 != **(_QWORD **)v10 )
    goto LABEL_23;
LABEL_8:
  v21 = v9 == v10;
LABEL_9:
  result = sub_22077B0(56);
  v15 = result;
  if ( result )
  {
    *(_QWORD *)result = a2;
    *(_QWORD *)(result + 40) = a8;
    *(_DWORD *)(result + 8) = a3;
    *(_DWORD *)(result + 12) = a4;
    *(_QWORD *)(result + 16) = a5;
    *(_QWORD *)(result + 24) = a6;
    *(_QWORD *)(result + 32) = a7;
    result = v21;
    *(_BYTE *)(v15 + 48) = v21;
    if ( v9 == v10 )
    {
      v22 = v15;
      result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 88LL))(a2);
      v15 = v22;
    }
  }
  v23[0] = v15;
  v16 = (char *)a1[2];
  if ( v16 == (char *)a1[3] )
  {
    result = sub_1C42910(a1 + 1, v16, v23);
    v15 = v23[0];
  }
  else
  {
    if ( v16 )
    {
      *(_QWORD *)v16 = v15;
      a1[2] += 8;
      return result;
    }
    a1[2] = 8;
  }
  if ( v15 )
    return j_j___libc_free_0(v15, 56);
  return result;
}

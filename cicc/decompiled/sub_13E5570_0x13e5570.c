// Function: sub_13E5570
// Address: 0x13e5570
//
__int64 __fastcall sub_13E5570(_QWORD *a1, __int64 a2)
{
  _BYTE *v3; // rsi
  __int64 v4; // r15
  __int64 result; // rax
  unsigned __int64 *v6; // rbx
  _QWORD *v7; // r13
  unsigned __int64 v8; // r14
  __int64 v9; // rax
  _QWORD *v10; // r15
  __int64 v11; // r8
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // rax
  _BOOL8 v15; // rdi
  __int64 v16; // [rsp+8h] [rbp-48h]
  unsigned __int64 *v17; // [rsp+10h] [rbp-40h]
  __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  v18[0] = a2;
  v3 = (_BYTE *)a1[28];
  if ( v3 == (_BYTE *)a1[29] )
  {
    sub_13E53E0((__int64)(a1 + 27), v3, v18);
    v4 = v18[0];
  }
  else
  {
    v4 = v18[0];
    if ( v3 )
    {
      *(_QWORD *)v3 = v18[0];
      v3 = (_BYTE *)a1[28];
    }
    a1[28] = v3 + 8;
  }
  result = *(_QWORD *)(v4 + 16);
  v6 = *(unsigned __int64 **)(v4 + 8);
  v17 = (unsigned __int64 *)result;
  if ( v6 != (unsigned __int64 *)result )
  {
    v7 = a1 + 21;
    while ( 1 )
    {
      v8 = *v6;
      v9 = sub_22077B0(48);
      *(_QWORD *)(v9 + 40) = v4;
      v10 = (_QWORD *)a1[22];
      v11 = v9;
      *(_QWORD *)(v9 + 32) = v8;
      if ( !v10 )
        break;
      while ( 1 )
      {
        v12 = v10[4];
        v13 = (_QWORD *)v10[3];
        if ( v8 < v12 )
          v13 = (_QWORD *)v10[2];
        if ( !v13 )
          break;
        v10 = v13;
      }
      if ( v8 < v12 )
      {
        if ( (_QWORD *)a1[23] != v10 )
          goto LABEL_18;
      }
      else if ( v12 >= v8 )
      {
        goto LABEL_15;
      }
LABEL_19:
      v15 = 1;
      if ( v7 != v10 )
        v15 = v8 < v10[4];
LABEL_21:
      ++v6;
      result = sub_220F040(v15, v11, v10, a1 + 21);
      ++a1[25];
      if ( v17 == v6 )
        return result;
LABEL_16:
      v4 = v18[0];
    }
    v10 = a1 + 21;
    if ( v7 == (_QWORD *)a1[23] )
    {
      v15 = 1;
      goto LABEL_21;
    }
LABEL_18:
    v16 = v11;
    v14 = sub_220EF80(v10);
    v11 = v16;
    if ( v8 <= *(_QWORD *)(v14 + 32) )
    {
LABEL_15:
      ++v6;
      result = j_j___libc_free_0(v11, 48);
      if ( v17 == v6 )
        return result;
      goto LABEL_16;
    }
    goto LABEL_19;
  }
  return result;
}

// Function: sub_2ABC990
// Address: 0x2abc990
//
__int64 __fastcall sub_2ABC990(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r9
  __int64 *v8; // r15
  __int64 *v9; // rbx
  char *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  char v14; // al
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 result; // rax
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rax
  unsigned __int64 v27; // rdi
  __int64 v28; // [rsp+8h] [rbp-B8h]
  __int64 v29; // [rsp+8h] [rbp-B8h]
  __int64 v30; // [rsp+8h] [rbp-B8h]
  __int64 v31[2]; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v32; // [rsp+20h] [rbp-A0h]
  int v33; // [rsp+28h] [rbp-98h]
  unsigned int v34; // [rsp+30h] [rbp-90h]
  unsigned __int64 v35; // [rsp+38h] [rbp-88h]
  __int64 v36; // [rsp+78h] [rbp-48h]

  v4 = a1;
  v8 = *(__int64 **)(a1 + 16);
  v9 = *(__int64 **)(a1 + 8);
  if ( v9 != v8 && !byte_500D208 )
  {
    if ( !LOBYTE(qword_500D340[17]) )
    {
      do
      {
LABEL_8:
        v18 = *v9++;
        result = sub_2ABC990(v18, a2, a3, a4);
      }
      while ( v8 != v9 );
      return result;
    }
    sub_31A4FD0(v31, a1, 1, a3, 0);
    v4 = a1;
    if ( (_DWORD)v35 == -1
      && (result = sub_F6E590(v36, a1, v19, v20, v21, a1), v4 = a1, !(_BYTE)result)
      && (_DWORD)v35 == -1
      || (v30 = v4,
          v22 = *(_QWORD *)(**(_QWORD **)(v4 + 32) + 72LL),
          result = sub_31A91F0(v31, v22, v4, 1),
          v4 = v30,
          !(_BYTE)result) )
    {
LABEL_7:
      v9 = *(__int64 **)(v4 + 8);
      v8 = *(__int64 **)(v4 + 16);
      if ( v8 == v9 )
        return result;
      goto LABEL_8;
    }
    if ( v33 )
    {
      if ( v33 != 1 )
      {
        result = sub_31A8E80(v31);
        v4 = v30;
        goto LABEL_7;
      }
    }
    else
    {
      sub_F6E5D0(v36, v22, v23, v24, v25, v30);
      v4 = v30;
    }
  }
  v28 = v4;
  sub_D33BC0((__int64)v31, v4);
  sub_D4E470(v31, a2);
  v14 = sub_DFEF30((__int64)v31, a2, v10, v11, v12, v13);
  v16 = v28;
  if ( v14 )
  {
    if ( v35 )
    {
      j_j___libc_free_0(v35);
      v16 = v28;
    }
    v29 = v16;
    result = sub_C7D6A0(v32, 16LL * v34, 8);
    v4 = v29;
    goto LABEL_7;
  }
  v26 = *(unsigned int *)(a4 + 8);
  if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
  {
    sub_C8D5F0(a4, (const void *)(a4 + 16), v26 + 1, 8u, v15, v28);
    v26 = *(unsigned int *)(a4 + 8);
    v16 = v28;
  }
  *(_QWORD *)(*(_QWORD *)a4 + 8 * v26) = v16;
  v27 = v35;
  ++*(_DWORD *)(a4 + 8);
  if ( v27 )
    j_j___libc_free_0(v27);
  return sub_C7D6A0(v32, 16LL * v34, 8);
}

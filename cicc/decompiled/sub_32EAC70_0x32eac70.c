// Function: sub_32EAC70
// Address: 0x32eac70
//
__int64 __fastcall sub_32EAC70(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v5; // r10
  __int64 v6; // rax
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v11; // rsi
  __int64 v12; // rcx
  unsigned int v13; // r13d
  __int64 v14; // rax
  __int64 v15; // r10
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // r15
  unsigned __int64 v19; // rbx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdi
  unsigned __int16 *v26; // rdx
  int v27; // r9d
  __int64 v28; // rax
  __int128 v29; // [rsp-20h] [rbp-90h]
  __int128 v30; // [rsp-10h] [rbp-80h]
  __int64 v31; // [rsp+8h] [rbp-68h]
  __int64 v32; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+18h] [rbp-58h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+18h] [rbp-58h]
  char v36; // [rsp+27h] [rbp-49h] BYREF
  __int64 v37; // [rsp+28h] [rbp-48h] BYREF
  __int64 v38; // [rsp+30h] [rbp-40h] BYREF
  int v39; // [rsp+38h] [rbp-38h]

  v5 = a2;
  v6 = 1;
  v9 = a1[1];
  if ( (_WORD)a4 != 1 )
  {
    if ( !(_WORD)a4 )
      return 0;
    v6 = (unsigned __int16)a4;
    if ( !*(_QWORD *)(v9 + 8LL * (unsigned __int16)a4 + 112) )
      return 0;
  }
  if ( *(_BYTE *)(v9 + 500 * v6 + 6636) )
    return 0;
  v11 = *(_QWORD *)(a2 + 80);
  v12 = *(_QWORD *)(*(_QWORD *)(v5 + 48) + 16LL * (unsigned int)a3 + 8);
  v13 = *(unsigned __int16 *)(*(_QWORD *)(v5 + 48) + 16LL * (unsigned int)a3);
  v38 = v11;
  v32 = v12;
  if ( v11 )
  {
    v31 = a5;
    v33 = v5;
    sub_B96E90((__int64)&v38, v11, 1);
    a5 = v31;
    v5 = v33;
  }
  v34 = v5;
  v39 = *(_DWORD *)(v5 + 72);
  v36 = 0;
  v14 = sub_32EA990(a1, v5, a3, a4, a5, &v36);
  v15 = v34;
  v16 = v14;
  v18 = v17;
  v19 = v14;
  if ( v14 )
  {
    if ( *(_DWORD *)(v14 + 24) != 328 )
    {
      v37 = v14;
      sub_32B3B20((__int64)(a1 + 71), &v37);
      v15 = v34;
      if ( *(int *)(v16 + 88) < 0 )
      {
        *(_DWORD *)(v16 + 88) = *((_DWORD *)a1 + 12);
        v28 = *((unsigned int *)a1 + 12);
        if ( v28 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
        {
          sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v28 + 1, 8u, v20, v21);
          v28 = *((unsigned int *)a1 + 12);
          v15 = v34;
        }
        *(_QWORD *)(a1[5] + 8 * v28) = v19;
        ++*((_DWORD *)a1 + 12);
      }
    }
    if ( v36 )
      sub_32EA6D0(a1, v15, v19);
    v22 = *a1;
    v23 = sub_33F7D60(*a1, v13, v32);
    v25 = v24;
    v26 = (unsigned __int16 *)(*(_QWORD *)(v19 + 48) + 16LL * (unsigned int)v18);
    *((_QWORD *)&v30 + 1) = v25;
    *(_QWORD *)&v30 = v23;
    *((_QWORD *)&v29 + 1) = v18;
    *(_QWORD *)&v29 = v16;
    result = sub_3406EB0(v22, 222, (unsigned int)&v38, *v26, *((_QWORD *)v26 + 1), v27, v29, v30);
  }
  else
  {
    result = 0;
  }
  if ( v38 )
  {
    v35 = result;
    sub_B91220((__int64)&v38, v38);
    return v35;
  }
  return result;
}

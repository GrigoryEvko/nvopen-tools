// Function: sub_2792D30
// Address: 0x2792d30
//
unsigned __int64 __fastcall sub_2792D30(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _DWORD *v9; // r13
  __int64 v10; // rax
  bool v11; // r12
  unsigned int v13; // esi
  int v14; // eax
  __int64 v15; // r12
  int v16; // eax
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // rdi
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  int v26; // edx
  __int64 v27; // rcx
  unsigned __int64 v28; // r8
  __int64 v29; // rdx
  __int64 v30; // [rsp+0h] [rbp-40h] BYREF
  __int64 v31[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a1 + 32;
  if ( (unsigned __int8)sub_278F8C0(a1 + 32, a2, &v30) )
  {
    v9 = (_DWORD *)(v30 + 56);
    v10 = *(unsigned int *)(v30 + 56);
    v11 = (_DWORD)v10 == 0;
    if ( (_DWORD)v10 )
      return ((unsigned __int64)v11 << 32) | v10;
    goto LABEL_9;
  }
  v13 = *(_DWORD *)(a1 + 56);
  v14 = *(_DWORD *)(a1 + 48);
  v15 = v30;
  ++*(_QWORD *)(a1 + 32);
  v16 = v14 + 1;
  v17 = 2 * v13;
  v31[0] = v15;
  if ( 4 * v16 >= 3 * v13 )
  {
    v13 *= 2;
  }
  else
  {
    v18 = v13 - *(_DWORD *)(a1 + 52) - v16;
    v19 = v13 >> 3;
    if ( (unsigned int)v18 > (unsigned int)v19 )
      goto LABEL_6;
  }
  sub_27929B0(v3, v13);
  sub_278F8C0(v3, a2, v31);
  v15 = v31[0];
  v16 = *(_DWORD *)(a1 + 48) + 1;
LABEL_6:
  *(_DWORD *)(a1 + 48) = v16;
  if ( *(_DWORD *)v15 != -1 )
    --*(_DWORD *)(a1 + 52);
  v9 = (_DWORD *)(v15 + 56);
  *(_DWORD *)v15 = *(_DWORD *)a2;
  *(_BYTE *)(v15 + 4) = *(_BYTE *)(a2 + 4);
  *(_QWORD *)(v15 + 8) = *(_QWORD *)(a2 + 8);
  sub_2789770(v15 + 16, a2 + 16, v18, v19, v17, v8);
  v20 = *(_QWORD *)(a2 + 48);
  *(_DWORD *)(v15 + 56) = 0;
  *(_QWORD *)(v15 + 48) = v20;
  v10 = *(unsigned int *)(v15 + 56);
  v11 = (_DWORD)v10 == 0;
  if ( !(_DWORD)v10 )
  {
LABEL_9:
    v21 = *(_QWORD *)(a1 + 80);
    if ( v21 == *(_QWORD *)(a1 + 88) )
    {
      sub_278FB90((unsigned __int64 *)(a1 + 72), *(_QWORD *)(a1 + 80), a2, v6, v7, v8);
    }
    else
    {
      if ( v21 )
      {
        *(_DWORD *)v21 = *(_DWORD *)a2;
        *(_BYTE *)(v21 + 4) = *(_BYTE *)(a2 + 4);
        *(_QWORD *)(v21 + 8) = *(_QWORD *)(a2 + 8);
        *(_QWORD *)(v21 + 16) = v21 + 32;
        *(_QWORD *)(v21 + 24) = 0x400000000LL;
        if ( *(_DWORD *)(a2 + 24) )
          sub_2789770(v21 + 16, a2 + 16, v5, v6, v7, v8);
        *(_QWORD *)(v21 + 48) = *(_QWORD *)(a2 + 48);
        v21 = *(_QWORD *)(a1 + 80);
      }
      *(_QWORD *)(a1 + 80) = v21 + 56;
    }
    v22 = *(_QWORD *)(a1 + 96);
    v23 = *(unsigned int *)(a1 + 208);
    v24 = (*(_QWORD *)(a1 + 104) - v22) >> 2;
    v25 = (unsigned int)(v23 + 1);
    if ( v25 > v24 )
    {
      v28 = (unsigned int)(2 * v23);
      if ( v28 > v24 )
      {
        sub_C17A60(a1 + 96, v28 - v24);
        v23 = *(unsigned int *)(a1 + 208);
        LODWORD(v25) = v23 + 1;
      }
      else if ( v28 < v24 )
      {
        v29 = v22 + 4 * v28;
        if ( *(_QWORD *)(a1 + 104) != v29 )
          *(_QWORD *)(a1 + 104) = v29;
      }
    }
    *v9 = v23;
    v26 = *(_DWORD *)(a1 + 64);
    *(_DWORD *)(a1 + 208) = v25;
    v27 = *(_QWORD *)(a1 + 96);
    *(_DWORD *)(a1 + 64) = v26 + 1;
    *(_DWORD *)(v27 + 4 * v23) = v26;
    v10 = (unsigned int)*v9;
  }
  return ((unsigned __int64)v11 << 32) | v10;
}

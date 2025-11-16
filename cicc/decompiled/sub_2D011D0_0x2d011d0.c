// Function: sub_2D011D0
// Address: 0x2d011d0
//
__int64 __fastcall sub_2D011D0(__int64 a1, unsigned __int64 a2)
{
  unsigned int v3; // eax
  unsigned int v4; // r13d
  __int64 v5; // rsi
  int v6; // eax
  int v7; // edx
  int v9; // r8d
  __int64 v10; // r9
  unsigned int v11; // ebx
  signed __int64 v12; // r15
  void *v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // r10
  _QWORD *v16; // rsi
  __int64 v17; // rdx
  _BYTE *v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rax
  unsigned int v21; // ebx
  unsigned int v22; // eax
  int v23; // edx
  unsigned __int64 v24; // r10
  int v25; // r8d
  int v26; // eax
  int v27; // r15d
  _BYTE *v28; // [rsp+0h] [rbp-50h]
  int v29; // [rsp+Ch] [rbp-44h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  int v31; // [rsp+18h] [rbp-38h]
  _QWORD *v32; // [rsp+18h] [rbp-38h]
  unsigned __int64 v33; // [rsp+18h] [rbp-38h]

  LOBYTE(v3) = sub_AC35E0(a2);
  v4 = v3;
  if ( (_BYTE)v3 )
  {
    v5 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *(_BYTE *)(*(_QWORD *)(v5 + 8) + 8LL) == 14 )
    {
      v27 = sub_2D00C30((unsigned int *)a1, (_BYTE *)v5);
      if ( v27 == (unsigned int)sub_2D00C30((unsigned int *)a1, (_BYTE *)a2) )
        return 0;
      v7 = v27;
    }
    else
    {
      v6 = sub_2D00C30((unsigned int *)a1, (_BYTE *)a2);
      v7 = *(_DWORD *)(a1 + 4);
      if ( v7 == v6 )
        return 0;
    }
    sub_2D00AD0((_QWORD *)a1, a2, v7);
    return v4;
  }
  if ( *(_WORD *)(a2 + 2) != 34 )
    return 0;
  v9 = sub_2D00C30((unsigned int *)a1, (_BYTE *)a2);
  v10 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( *(_BYTE *)(*(_QWORD *)(v10 + 8) + 8LL) == 14 )
  {
    v11 = (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) - 1;
    if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == 1 )
    {
      v12 = 0;
      v15 = 0;
    }
    else
    {
      v12 = 8LL * v11;
      v30 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      v31 = v9;
      v13 = (void *)sub_22077B0(v12);
      v14 = memset(v13, 0, v12);
      v9 = v31;
      v10 = v30;
      v15 = v14;
    }
    v16 = v15;
    LODWORD(v17) = 0;
    while ( (_DWORD)v17 != v11 )
    {
      ++v16;
      v17 = (unsigned int)(v17 + 1);
      v18 = *(_BYTE **)(a2 + 32 * (v17 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
      *(v16 - 1) = v18;
      if ( *v18 != 17 )
      {
        v33 = (unsigned __int64)v15;
        v23 = *(_DWORD *)(a1 + 4);
        goto LABEL_17;
      }
    }
    v29 = v9;
    v19 = *(_QWORD *)(a1 + 248);
    v28 = (_BYTE *)v10;
    v32 = v15;
    v20 = sub_BB5290(a2);
    v21 = sub_AE54E0(v19, v20, v32, v12 >> 3);
    v22 = sub_2D00C30((unsigned int *)a1, v28);
    v23 = *(_DWORD *)a1;
    v24 = (unsigned __int64)v32;
    v25 = v29;
    if ( v22 != *(_DWORD *)a1 )
    {
      v26 = sub_2D00850(a1, v21, v22);
      v24 = (unsigned __int64)v32;
      v25 = v29;
      v23 = v26;
    }
    if ( v23 != v25 )
    {
      v33 = v24;
LABEL_17:
      v4 = 1;
      sub_2D00AD0((_QWORD *)a1, a2, v23);
      v24 = v33;
    }
    if ( v24 )
      j_j___libc_free_0(v24);
  }
  return v4;
}

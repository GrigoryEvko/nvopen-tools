// Function: sub_2E01820
// Address: 0x2e01820
//
__int64 __fastcall sub_2E01820(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // rsi
  __int64 v6; // rax
  __int64 *v7; // r14
  __int64 v8; // rax
  __int64 *v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rsi
  unsigned int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // r9
  __int64 v16; // rax
  _QWORD *v17; // r13
  _QWORD *v18; // rax
  _QWORD *v19; // rbx
  unsigned __int64 v20; // r13
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  unsigned int v23; // esi
  __int64 v24; // rdi
  __int64 v25; // r8
  int v26; // r10d
  unsigned int v27; // edx
  __int64 *v28; // rax
  __int64 v29; // rcx
  __int64 result; // rax
  int v31; // eax
  __int16 v32; // dx
  int v33; // edx
  int v34; // eax
  int v35; // r10d
  int v36; // ecx
  __int64 v37; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v38; // [rsp+18h] [rbp-28h] BYREF

  v5 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = *(unsigned int *)(a1 + 24);
  v7 = *(__int64 **)(v5 + 16);
  v37 = a1;
  v8 = *(_QWORD *)(*(_QWORD *)(a3 + 152) + 16 * v6);
  if ( !v7 )
  {
    while ( v8 != v5 )
    {
      v5 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
      v10 = *(__int64 **)(v5 + 16);
      if ( v10 )
        goto LABEL_24;
    }
    v11 = *(unsigned int *)(a4 + 24);
    v12 = *(_QWORD *)(a4 + 8);
    if ( (_DWORD)v11 )
    {
      v13 = (v11 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v14 = (__int64 *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( a1 == *v14 )
      {
LABEL_7:
        if ( v14 != (__int64 *)(v12 + 16 * v11) )
        {
          v16 = v14[1];
          if ( !v16 )
            BUG();
          if ( (*(_BYTE *)v16 & 4) == 0 && (*(_BYTE *)(v16 + 44) & 8) != 0 )
          {
            do
              v16 = *(_QWORD *)(v16 + 8);
            while ( (*(_BYTE *)(v16 + 44) & 8) != 0 );
          }
          v17 = *(_QWORD **)(v16 + 8);
LABEL_11:
          v18 = (_QWORD *)sub_2E312E0(a1, v17, 0, 1);
          v19 = v18;
          if ( v17 == v18 )
            return (__int64)v19;
          v20 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v20 )
            BUG();
          v21 = *(_QWORD *)v20;
          if ( (*(_QWORD *)v20 & 4) == 0 && (*(_BYTE *)(v20 + 44) & 4) != 0 )
          {
            while ( 1 )
            {
              v22 = v21 & 0xFFFFFFFFFFFFFFF8LL;
              v20 = v22;
              if ( (*(_BYTE *)(v22 + 44) & 4) == 0 )
                break;
              v21 = *(_QWORD *)v22;
            }
          }
          v23 = *(_DWORD *)(a4 + 24);
          if ( v23 )
          {
            v24 = v37;
            v25 = *(_QWORD *)(a4 + 8);
            v26 = 1;
            v27 = (v23 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
            v28 = (__int64 *)(v25 + 16LL * v27);
            v29 = *v28;
            if ( v37 == *v28 )
            {
LABEL_20:
              v28[1] = v20;
              return (__int64)v19;
            }
            while ( v29 != -4096 )
            {
              if ( v29 == -8192 && !v7 )
                v7 = v28;
              v27 = (v23 - 1) & (v26 + v27);
              v28 = (__int64 *)(v25 + 16LL * v27);
              v29 = *v28;
              if ( v37 == *v28 )
                goto LABEL_20;
              ++v26;
            }
            v36 = *(_DWORD *)(a4 + 16);
            if ( v7 )
              v28 = v7;
            ++*(_QWORD *)a4;
            v33 = v36 + 1;
            v38 = v28;
            if ( 4 * (v36 + 1) < 3 * v23 )
            {
              if ( v23 - *(_DWORD *)(a4 + 20) - v33 > v23 >> 3 )
                goto LABEL_50;
              goto LABEL_49;
            }
          }
          else
          {
            ++*(_QWORD *)a4;
            v38 = 0;
          }
          v23 *= 2;
LABEL_49:
          sub_2E01640(a4, v23);
          sub_2DFA9A0(a4, &v37, &v38);
          v24 = v37;
          v33 = *(_DWORD *)(a4 + 16) + 1;
          v28 = v38;
LABEL_50:
          *(_DWORD *)(a4 + 16) = v33;
          if ( *v28 != -4096 )
            --*(_DWORD *)(a4 + 20);
          *v28 = v24;
          v28[1] = 0;
          goto LABEL_20;
        }
      }
      else
      {
        v34 = 1;
        while ( v15 != -4096 )
        {
          v35 = v34 + 1;
          v13 = (v11 - 1) & (v34 + v13);
          v14 = (__int64 *)(v12 + 16LL * v13);
          v15 = *v14;
          if ( a1 == *v14 )
            goto LABEL_7;
          v34 = v35;
        }
      }
    }
    v17 = *(_QWORD **)(a1 + 56);
    goto LABEL_11;
  }
  v10 = v7;
LABEL_24:
  v31 = *((_DWORD *)v10 + 11);
  if ( (v31 & 4) != 0 || (v31 & 8) == 0 )
  {
    if ( (*(_QWORD *)(v10[2] + 24) & 0x200LL) == 0 )
      goto LABEL_38;
LABEL_26:
    result = sub_2E313E0(a1, v5, a3, a4, a5);
    a1 = v37;
    goto LABEL_27;
  }
  v5 = 512;
  a1 = v37;
  if ( (unsigned __int8)sub_2E88A90(v10, 512, 1) )
    goto LABEL_26;
LABEL_38:
  if ( (*(_BYTE *)v10 & 4) != 0 )
  {
    result = v10[1];
  }
  else
  {
    while ( (*((_BYTE *)v10 + 44) & 8) != 0 )
      v10 = (__int64 *)v10[1];
    result = v10[1];
  }
LABEL_27:
  while ( a1 + 48 != result )
  {
    v32 = *(_WORD *)(result + 68);
    if ( (unsigned __int16)(v32 - 14) > 4u && v32 != 24 )
      break;
    if ( (*(_BYTE *)result & 4) == 0 && (*(_BYTE *)(result + 44) & 8) != 0 )
    {
      do
        result = *(_QWORD *)(result + 8);
      while ( (*(_BYTE *)(result + 44) & 8) != 0 );
    }
    result = *(_QWORD *)(result + 8);
  }
  return result;
}

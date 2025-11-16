// Function: sub_2E64280
// Address: 0x2e64280
//
void __fastcall sub_2E64280(__int64 a1, __int64 a2)
{
  __int64 v4; // r8
  __int64 v5; // rdx
  unsigned int v6; // eax
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  _QWORD *v10; // r13
  __int64 *v11; // rax
  __int64 v12; // r11
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 v17; // r10
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // rdi
  __int64 v20; // r13
  __int64 v21; // rdx
  unsigned int v22; // eax
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 *v26; // rax
  __int64 v27; // r10
  __int64 v28; // r11
  __int64 *v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  unsigned __int64 v33; // r12
  unsigned __int64 v34; // rdi
  __int64 *v35; // rsi
  __int64 v36; // rcx
  _QWORD *v37; // r8
  __int64 v38; // rdi
  __int64 *v39; // rax
  __int64 *v40; // rdx
  __int64 v41; // rcx
  __int64 v42[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_QWORD *)(a1 + 544);
  if ( v4 && !*(_BYTE *)(a1 + 664) )
  {
    if ( a2 )
    {
      v5 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
      v6 = *(_DWORD *)(a2 + 24) + 1;
    }
    else
    {
      v5 = 0;
      v6 = 0;
    }
    if ( v6 < *(_DWORD *)(v4 + 32) )
    {
      v7 = (__int64 *)(8 * v5 + *(_QWORD *)(v4 + 24));
      v8 = *v7;
      if ( *v7 )
      {
        *(_BYTE *)(v4 + 112) = 0;
        v9 = *(_QWORD *)(v8 + 8);
        v42[0] = v8;
        if ( v9 )
        {
          v10 = *(_QWORD **)(v9 + 24);
          v11 = sub_2E641C0(v10, (__int64)&v10[*(unsigned int *)(v9 + 32)], v42);
          v13 = (_QWORD *)((char *)v10 + v12 - 8);
          v14 = *v11;
          *v11 = *v13;
          *v13 = v14;
          --*(_DWORD *)(v15 + 32);
          v7 = (__int64 *)(v17 + *(_QWORD *)(v16 + 24));
        }
        v18 = *v7;
        *v7 = 0;
        if ( v18 )
        {
          v19 = *(_QWORD *)(v18 + 24);
          if ( v19 != v18 + 40 )
            _libc_free(v19);
          j_j___libc_free_0(v18);
        }
      }
    }
  }
  v20 = *(_QWORD *)(a1 + 552);
  if ( v20 && !*(_BYTE *)(a1 + 665) )
  {
    if ( a2 )
    {
      v21 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
      v22 = *(_DWORD *)(a2 + 24) + 1;
    }
    else
    {
      v21 = 0;
      v22 = 0;
    }
    if ( v22 < *(_DWORD *)(v20 + 56) )
    {
      v23 = (__int64 *)(8 * v21 + *(_QWORD *)(v20 + 48));
      v24 = *v23;
      if ( *v23 )
      {
        *(_BYTE *)(v20 + 136) = 0;
        v25 = *(_QWORD *)(v24 + 8);
        v42[0] = v24;
        if ( v25 )
        {
          v26 = sub_2E641C0(*(_QWORD **)(v25 + 24), *(_QWORD *)(v25 + 24) + 8LL * *(unsigned int *)(v25 + 32), v42);
          v29 = (__int64 *)(v28 + v27 - 8);
          v30 = *v26;
          *v26 = *v29;
          *v29 = v30;
          --*(_DWORD *)(v31 + 32);
          v23 = (__int64 *)(v32 + *(_QWORD *)(v20 + 48));
        }
        v33 = *v23;
        *v23 = 0;
        if ( v33 )
        {
          v34 = *(_QWORD *)(v33 + 24);
          if ( v34 != v33 + 40 )
            _libc_free(v34);
          j_j___libc_free_0(v33);
        }
        v35 = *(__int64 **)v20;
        v36 = 8LL * *(unsigned int *)(v20 + 8);
        v37 = (_QWORD *)(*(_QWORD *)v20 + v36);
        v38 = v36 >> 3;
        if ( v36 >> 5 )
        {
          v39 = *(__int64 **)v20;
          while ( a2 != *v39 )
          {
            if ( a2 == v39[1] )
            {
              ++v39;
              goto LABEL_32;
            }
            if ( a2 == v39[2] )
            {
              v39 += 2;
              goto LABEL_32;
            }
            if ( a2 == v39[3] )
            {
              v39 += 3;
              goto LABEL_32;
            }
            v39 += 4;
            if ( &v35[4 * (v36 >> 5)] == v39 )
            {
              v38 = v37 - v39;
              goto LABEL_41;
            }
          }
          goto LABEL_32;
        }
        v39 = *(__int64 **)v20;
LABEL_41:
        if ( v38 != 2 )
        {
          if ( v38 != 3 )
          {
            if ( v38 != 1 )
              return;
LABEL_44:
            if ( a2 != *v39 )
              return;
            goto LABEL_32;
          }
          if ( a2 == *v39 )
          {
LABEL_32:
            if ( v37 != v39 )
            {
              v40 = &v35[(unsigned __int64)v36 / 8 - 1];
              v41 = *v39;
              *v39 = *v40;
              *v40 = v41;
              --*(_DWORD *)(v20 + 8);
            }
            return;
          }
          ++v39;
        }
        if ( a2 != *v39 )
        {
          ++v39;
          goto LABEL_44;
        }
        goto LABEL_32;
      }
    }
  }
}

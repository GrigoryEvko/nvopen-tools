// Function: sub_35059C0
// Address: 0x35059c0
//
void __fastcall sub_35059C0(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  __int64 v5; // rdx
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  unsigned __int8 **v8; // rcx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // r12
  _QWORD *v15; // r13
  __int64 v16; // rbx
  __int64 v17; // r15
  char v18; // si
  __int64 *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rbx
  __int64 v22; // r12
  char v23; // si
  __int64 *v24; // rax

  ++*(_QWORD *)a3;
  if ( !*(_BYTE *)(a3 + 28) )
  {
    v4 = 4 * (*(_DWORD *)(a3 + 20) - *(_DWORD *)(a3 + 24));
    v5 = *(unsigned int *)(a3 + 16);
    if ( v4 < 0x20 )
      v4 = 32;
    if ( (unsigned int)v5 > v4 )
    {
      sub_C8C990(a3, a2);
      goto LABEL_7;
    }
    memset(*(void **)(a3 + 8), -1, 8 * v5);
  }
  *(_QWORD *)(a3 + 20) = 0;
LABEL_7:
  if ( a2 )
  {
    v6 = *(_BYTE *)(a2 - 16);
    if ( (v6 & 2) != 0 )
    {
      v7 = *(_DWORD *)(a2 - 24) == 2 ? *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL) : 0LL;
      v8 = *(unsigned __int8 ***)(a2 - 32);
    }
    else
    {
      v20 = a2 - 16;
      v7 = ((*(_WORD *)(a2 - 16) >> 6) & 0xF) == 2 ? *(_QWORD *)(v20 - 8LL * ((v6 >> 2) & 0xF) + 8) : 0LL;
      v8 = (unsigned __int8 **)(v20 - 8LL * ((v6 >> 2) & 0xF));
    }
    v9 = sub_35057B0(a1, *v8, v7);
    if ( v9 )
    {
      if ( a1[28] == v9 )
      {
        v21 = *(_QWORD *)(*a1 + 328LL);
        v22 = *a1 + 320LL;
        if ( v21 != v22 )
        {
          v23 = *(_BYTE *)(a3 + 28);
          if ( !v23 )
            goto LABEL_43;
LABEL_37:
          v24 = *(__int64 **)(a3 + 8);
          v11 = *(unsigned int *)(a3 + 20);
          v10 = &v24[v11];
          if ( v24 == v10 )
          {
LABEL_45:
            if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 16) )
            {
LABEL_43:
              while ( 1 )
              {
                sub_C8CC70(a3, v21, (__int64)v10, v11, v12, v13);
                v21 = *(_QWORD *)(v21 + 8);
                v23 = *(_BYTE *)(a3 + 28);
                if ( v21 == v22 )
                  break;
LABEL_42:
                if ( v23 )
                  goto LABEL_37;
              }
            }
            else
            {
              v11 = (unsigned int)(v11 + 1);
              *(_DWORD *)(a3 + 20) = v11;
              *v10 = v21;
              v23 = *(_BYTE *)(a3 + 28);
              ++*(_QWORD *)a3;
              v21 = *(_QWORD *)(v21 + 8);
              if ( v21 != v22 )
                goto LABEL_42;
            }
          }
          else
          {
            while ( v21 != *v24 )
            {
              if ( v10 == ++v24 )
                goto LABEL_45;
            }
            v21 = *(_QWORD *)(v21 + 8);
            if ( v21 != v22 )
              goto LABEL_42;
          }
        }
      }
      else
      {
        v14 = *(_QWORD **)(v9 + 80);
        v15 = &v14[2 * *(unsigned int *)(v9 + 88)];
        if ( v15 != v14 )
        {
          while ( 1 )
          {
            v16 = *(_QWORD *)(*v14 + 24LL);
            v17 = *(_QWORD *)(*(_QWORD *)(v14[1] + 24LL) + 8LL);
            if ( v16 != v17 )
              break;
LABEL_23:
            v14 += 2;
            if ( v15 == v14 )
              return;
          }
          v18 = *(_BYTE *)(a3 + 28);
          while ( 1 )
          {
            while ( !v18 )
            {
LABEL_26:
              sub_C8CC70(a3, v16, (__int64)v10, v11, v12, v13);
              v16 = *(_QWORD *)(v16 + 8);
              v18 = *(_BYTE *)(a3 + 28);
              if ( v16 == v17 )
                goto LABEL_23;
            }
            v19 = *(__int64 **)(a3 + 8);
            v11 = *(unsigned int *)(a3 + 20);
            v10 = &v19[v11];
            if ( v19 == v10 )
            {
LABEL_28:
              if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 16) )
                goto LABEL_26;
              v11 = (unsigned int)(v11 + 1);
              *(_DWORD *)(a3 + 20) = v11;
              *v10 = v16;
              v18 = *(_BYTE *)(a3 + 28);
              ++*(_QWORD *)a3;
              v16 = *(_QWORD *)(v16 + 8);
              if ( v16 == v17 )
                goto LABEL_23;
            }
            else
            {
              while ( v16 != *v19 )
              {
                if ( v10 == ++v19 )
                  goto LABEL_28;
              }
              v16 = *(_QWORD *)(v16 + 8);
              if ( v16 == v17 )
                goto LABEL_23;
            }
          }
        }
      }
    }
  }
}

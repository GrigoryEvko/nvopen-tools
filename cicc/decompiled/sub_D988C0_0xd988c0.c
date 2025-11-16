// Function: sub_D988C0
// Address: 0xd988c0
//
__int64 __fastcall sub_D988C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int *v5; // rsi
  __int64 result; // rax
  __int64 v9; // r14
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int v14; // r13d
  int v15; // esi
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r10
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // r13
  _QWORD *v22; // rax
  __int64 v23; // rax
  unsigned int v24; // r13d
  __int64 *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rax
  int v28; // eax
  int v29; // eax
  int v30; // eax
  int v31; // esi
  __int64 v32; // rax
  const void *v33; // [rsp+8h] [rbp-88h]
  __int64 v34; // [rsp+18h] [rbp-78h]
  int v36; // [rsp+28h] [rbp-68h]
  __int64 v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+28h] [rbp-68h]
  __int64 v39; // [rsp+28h] [rbp-68h]
  void *v40; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v41[2]; // [rsp+38h] [rbp-58h] BYREF
  unsigned __int64 v42; // [rsp+48h] [rbp-48h]

  v5 = (unsigned int *)(a2 + 16);
  result = *(v5 - 2);
  v33 = v5;
  while ( (_DWORD)result )
  {
    v9 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * (unsigned int)result - 8);
    *(_DWORD *)(a2 + 8) = result - 1;
    if ( !sub_D97040(a1, *(_QWORD *)(v9 + 8)) )
    {
      if ( *(_BYTE *)v9 != 85 )
        goto LABEL_4;
      v27 = *(_QWORD *)(v9 - 32);
      if ( !v27 || *(_BYTE *)v27 || *(_QWORD *)(v27 + 24) != *(_QWORD *)(v9 + 80) || (*(_BYTE *)(v27 + 33) & 0x20) == 0 )
        goto LABEL_4;
      v28 = *(_DWORD *)(v27 + 36);
      if ( v28 != 312 )
      {
        switch ( v28 )
        {
          case 333:
          case 339:
          case 360:
          case 369:
          case 372:
            goto LABEL_6;
          default:
            goto LABEL_4;
        }
        goto LABEL_4;
      }
    }
LABEL_6:
    v36 = *(_DWORD *)(a1 + 152);
    if ( v36 )
    {
      v34 = *(_QWORD *)(a1 + 136);
      sub_D982A0(&v40, -4096, 0);
      v11 = v42;
      v14 = ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4);
      v10 = (unsigned int)(v36 - 1);
      v15 = v10 & v14;
      v16 = v34 + 48LL * ((unsigned int)v10 & v14);
      v17 = *(_QWORD *)(v16 + 24);
      if ( v9 == v17 )
      {
LABEL_8:
        v40 = &unk_49DB368;
        if ( v42 && v42 != -4096 && v42 != -8192 )
        {
          v37 = v16;
          sub_BD60C0(v41);
          v16 = v37;
        }
        v10 = *(_QWORD *)(a1 + 136) + 48LL * *(unsigned int *)(a1 + 152);
        if ( v16 != v10 )
        {
          v38 = v16;
          sub_D98440(a1, *(_QWORD *)(v16 + 24));
          v18 = *(_QWORD *)(v38 + 40);
          v19 = *(unsigned int *)(a4 + 8);
          v11 = *(unsigned int *)(a4 + 12);
          if ( v19 + 1 > v11 )
          {
            v39 = *(_QWORD *)(v38 + 40);
            sub_C8D5F0(a4, (const void *)(a4 + 16), v19 + 1, 8u, v12, v13);
            v18 = v39;
            v19 = *(unsigned int *)(a4 + 8);
          }
          v10 = *(_QWORD *)a4;
          *(_QWORD *)(*(_QWORD *)a4 + 8 * v19) = v18;
          ++*(_DWORD *)(a4 + 8);
          if ( *(_BYTE *)v9 == 84 )
          {
            v10 = *(unsigned int *)(a1 + 768);
            v11 = *(_QWORD *)(a1 + 752);
            if ( (_DWORD)v10 )
            {
              v10 = (unsigned int)(v10 - 1);
              v24 = v10 & v14;
              v25 = (__int64 *)(v11 + 16LL * v24);
              v26 = *v25;
              if ( v9 == *v25 )
              {
LABEL_33:
                *v25 = -8192;
                --*(_DWORD *)(a1 + 760);
                ++*(_DWORD *)(a1 + 764);
              }
              else
              {
                v30 = 1;
                while ( v26 != -4096 )
                {
                  v31 = v30 + 1;
                  v24 = v10 & (v30 + v24);
                  v25 = (__int64 *)(v11 + 16LL * v24);
                  v26 = *v25;
                  if ( v9 == *v25 )
                    goto LABEL_33;
                  v30 = v31;
                }
              }
            }
          }
        }
      }
      else
      {
        v29 = 1;
        while ( v42 != v17 )
        {
          v12 = (unsigned int)(v29 + 1);
          v32 = (unsigned int)v10 & (v15 + v29);
          v15 = v32;
          v16 = v34 + 48 * v32;
          v17 = *(_QWORD *)(v16 + 24);
          if ( v9 == v17 )
            goto LABEL_8;
          v29 = v12;
        }
        if ( v42 && v42 != -4096 && v42 != -8192 )
        {
          v40 = &unk_49DB368;
          sub_BD60C0(v41);
        }
      }
    }
    v20 = *(_QWORD *)(v9 + 16);
    if ( v20 )
    {
      v21 = *(_QWORD *)(v20 + 24);
      if ( *(_BYTE *)(a3 + 28) )
      {
LABEL_18:
        v22 = *(_QWORD **)(a3 + 8);
        v11 = *(unsigned int *)(a3 + 20);
        v10 = (__int64)&v22[v11];
        if ( v22 == (_QWORD *)v10 )
          goto LABEL_29;
        while ( v21 != *v22 )
        {
          if ( (_QWORD *)v10 == ++v22 )
          {
LABEL_29:
            if ( (unsigned int)v11 < *(_DWORD *)(a3 + 16) )
            {
              *(_DWORD *)(a3 + 20) = v11 + 1;
              *(_QWORD *)v10 = v21;
              ++*(_QWORD *)a3;
              goto LABEL_25;
            }
            goto LABEL_24;
          }
        }
        goto LABEL_22;
      }
      while ( 1 )
      {
LABEL_24:
        sub_C8CC70(a3, v21, v10, v11, v12, v13);
        if ( (_BYTE)v10 )
        {
LABEL_25:
          v23 = *(unsigned int *)(a2 + 8);
          v11 = *(unsigned int *)(a2 + 12);
          if ( v23 + 1 > v11 )
          {
            sub_C8D5F0(a2, v33, v23 + 1, 8u, v12, v13);
            v23 = *(unsigned int *)(a2 + 8);
          }
          v10 = *(_QWORD *)a2;
          *(_QWORD *)(*(_QWORD *)a2 + 8 * v23) = v21;
          ++*(_DWORD *)(a2 + 8);
          v20 = *(_QWORD *)(v20 + 8);
          if ( !v20 )
            break;
        }
        else
        {
LABEL_22:
          v20 = *(_QWORD *)(v20 + 8);
          if ( !v20 )
            break;
        }
        v21 = *(_QWORD *)(v20 + 24);
        if ( *(_BYTE *)(a3 + 28) )
          goto LABEL_18;
      }
    }
LABEL_4:
    result = *(unsigned int *)(a2 + 8);
  }
  return result;
}

// Function: sub_14D04F0
// Address: 0x14d04f0
//
void __fastcall sub_14D04F0(__int64 a1, __int64 a2, __int64 a3)
{
  bool v5; // zf
  __int64 v6; // r14
  __int64 v7; // rbx
  char v8; // dl
  __int64 v9; // rbx
  _QWORD *v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rsi
  _QWORD *v13; // r12
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 *v16; // rdi
  unsigned int v17; // r11d
  __int64 *v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rdi
  _QWORD *v21; // rsi
  __int64 v22; // [rsp+18h] [rbp-208h]
  __int64 v23; // [rsp+20h] [rbp-200h]
  __int64 v24; // [rsp+28h] [rbp-1F8h]
  unsigned __int64 v25[2]; // [rsp+30h] [rbp-1F0h] BYREF
  _BYTE v26[128]; // [rsp+40h] [rbp-1E0h] BYREF
  __int64 v27; // [rsp+C0h] [rbp-160h] BYREF
  _BYTE *v28; // [rsp+C8h] [rbp-158h]
  _BYTE *v29; // [rsp+D0h] [rbp-150h]
  __int64 v30; // [rsp+D8h] [rbp-148h]
  int v31; // [rsp+E0h] [rbp-140h]
  _BYTE v32[312]; // [rsp+E8h] [rbp-138h] BYREF

  v28 = v32;
  v5 = *(_BYTE *)(a2 + 184) == 0;
  v29 = v32;
  v25[0] = (unsigned __int64)v26;
  v27 = 0;
  v30 = 32;
  v31 = 0;
  v25[1] = 0x1000000000LL;
  if ( v5 )
    sub_14CDF70(a2);
  v6 = *(_QWORD *)(a2 + 8);
  v7 = 32LL * *(unsigned int *)(a2 + 16);
  v22 = a1 + 56;
  v24 = v6 + v7;
  if ( v6 + v7 != v6 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v6 + 16);
      if ( !v9 )
        goto LABEL_6;
      v10 = *(_QWORD **)(a1 + 72);
      v11 = *(_QWORD **)(a1 + 64);
      v12 = *(_QWORD *)(v9 + 40);
      if ( v10 == v11 )
      {
        v13 = &v11[*(unsigned int *)(a1 + 84)];
        if ( v11 == v13 )
        {
          v21 = *(_QWORD **)(a1 + 64);
        }
        else
        {
          do
          {
            if ( v12 == *v11 )
              break;
            ++v11;
          }
          while ( v13 != v11 );
          v21 = v13;
        }
      }
      else
      {
        v23 = *(_QWORD *)(v9 + 40);
        v13 = &v10[*(unsigned int *)(a1 + 80)];
        v11 = (_QWORD *)sub_16CC9F0(v22, v12);
        if ( v23 == *v11 )
        {
          v19 = *(_QWORD *)(a1 + 72);
          if ( v19 == *(_QWORD *)(a1 + 64) )
            v20 = *(unsigned int *)(a1 + 84);
          else
            v20 = *(unsigned int *)(a1 + 80);
          v21 = (_QWORD *)(v19 + 8 * v20);
        }
        else
        {
          v14 = *(_QWORD *)(a1 + 72);
          if ( v14 != *(_QWORD *)(a1 + 64) )
          {
            v11 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a1 + 80));
            goto LABEL_12;
          }
          v11 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a1 + 84));
          v21 = v11;
        }
      }
      while ( v21 != v11 && *v11 >= 0xFFFFFFFFFFFFFFFELL )
        ++v11;
LABEL_12:
      if ( v11 == v13 )
        goto LABEL_6;
      v15 = *(__int64 **)(a3 + 8);
      if ( *(__int64 **)(a3 + 16) == v15 )
      {
        v16 = &v15[*(unsigned int *)(a3 + 28)];
        v17 = *(_DWORD *)(a3 + 28);
        if ( v15 != v16 )
        {
          v18 = 0;
          while ( v9 != *v15 )
          {
            if ( *v15 == -2 )
              v18 = v15;
            if ( v16 == ++v15 )
            {
              if ( !v18 )
                goto LABEL_41;
              *v18 = v9;
              --*(_DWORD *)(a3 + 32);
              ++*(_QWORD *)a3;
              goto LABEL_22;
            }
          }
          goto LABEL_6;
        }
LABEL_41:
        if ( v17 < *(_DWORD *)(a3 + 24) )
        {
          *(_DWORD *)(a3 + 28) = v17 + 1;
          *v16 = v9;
          ++*(_QWORD *)a3;
          goto LABEL_22;
        }
      }
      sub_16CCBA0(a3, v9);
      if ( v8 )
      {
LABEL_22:
        v6 += 32;
        sub_14D01A0(v9, (__int64)&v27, (__int64)v25);
        if ( v24 == v6 )
          break;
      }
      else
      {
LABEL_6:
        v6 += 32;
        if ( v24 == v6 )
          break;
      }
    }
  }
  sub_14D02F0((__int64)&v27, (__int64)v25, a3);
  if ( (_BYTE *)v25[0] != v26 )
    _libc_free(v25[0]);
  if ( v29 != v28 )
    _libc_free((unsigned __int64)v29);
}

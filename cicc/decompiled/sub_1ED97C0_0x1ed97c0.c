// Function: sub_1ED97C0
// Address: 0x1ed97c0
//
__int64 __fastcall sub_1ED97C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v5; // rbx
  unsigned __int64 v6; // rbx
  __int64 v7; // rax
  _DWORD *v9; // rax
  int v10; // r12d
  __int64 v11; // r15
  unsigned int v12; // eax
  unsigned __int64 v13; // rdx
  __int64 v14; // r9
  __int64 v15; // r10
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 *v18; // rdx
  __int64 v19; // rcx
  unsigned int v20; // r14d
  __int64 v21; // rcx
  __int64 v22; // r14
  int v23; // eax
  __int64 *v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rax
  _QWORD *v28; // rsi
  _QWORD *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+28h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 8);
  v32 = a2;
  if ( (v5 & 6) != 0 )
  {
    while ( 1 )
    {
      v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v6 )
        BUG();
      v7 = *(_QWORD *)(v6 + 16);
      if ( **(_WORD **)(v7 + 16) != 15 )
        return v32;
      v9 = *(_DWORD **)(v7 + 32);
      if ( (*v9 & 0xFFF00) != 0 )
        return v32;
      if ( (v9[10] & 0xFFF00) != 0 )
        return v32;
      v10 = v9[12];
      if ( v10 >= 0 )
        return v32;
      v11 = *(_QWORD *)(a1 + 40);
      v12 = v10 & 0x7FFFFFFF;
      v13 = *(unsigned int *)(v11 + 408);
      v14 = v10 & 0x7FFFFFFF;
      v15 = 8 * v14;
      if ( (v10 & 0x7FFFFFFFu) >= (unsigned int)v13 )
        break;
      v16 = *(_QWORD *)(*(_QWORD *)(v11 + 400) + 8LL * v12);
      if ( !v16 )
        break;
LABEL_10:
      if ( *(_BYTE *)(a1 + 20) )
      {
        v17 = *(_QWORD *)(v16 + 104);
        if ( v17 )
        {
          v22 = 0;
          while ( 1 )
          {
            while ( 1 )
            {
              v23 = *(_DWORD *)(v17 + 112);
              if ( *(_DWORD *)(a1 + 12) )
                v23 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 56) + 128LL))(*(_QWORD *)(a1 + 56));
              if ( (*(_DWORD *)(a1 + 16) & v23) != 0 )
              {
                v24 = (__int64 *)sub_1DB3C70((__int64 *)v17, v6);
                v25 = *(_QWORD *)v17 + 24LL * *(unsigned int *)(v17 + 8);
                if ( v24 != (__int64 *)v25 )
                  break;
              }
LABEL_21:
              v17 = *(_QWORD *)(v17 + 104);
              if ( !v17 )
                goto LABEL_32;
            }
            v26 = 0;
            if ( (*(_DWORD *)((*v24 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v24 >> 1) & 3) <= *(_DWORD *)(v6 + 24) )
            {
              v26 = v24[2];
              if ( (v6 != (v24[1] & 0xFFFFFFFFFFFFFFF8LL) || (__int64 *)v25 != v24 + 3) && v6 == *(_QWORD *)(v26 + 8) )
                v26 = 0;
            }
            if ( v22 )
            {
              if ( v26 && v26 != v22 )
                return v32;
              goto LABEL_21;
            }
            v17 = *(_QWORD *)(v17 + 104);
            v22 = v26;
            if ( !v17 )
            {
LABEL_32:
              v32 = v22;
              goto LABEL_33;
            }
          }
        }
      }
      v18 = (__int64 *)sub_1DB3C70((__int64 *)v16, v6);
      v19 = *(_QWORD *)v16 + 24LL * *(unsigned int *)(v16 + 8);
      if ( v18 == (__int64 *)v19
        || (*(_DWORD *)((*v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v18 >> 1) & 3) > *(_DWORD *)(v6 + 24) )
      {
        return 0;
      }
      v32 = v18[2];
      if ( (v6 != (v18[1] & 0xFFFFFFFFFFFFFFF8LL) || (__int64 *)v19 != v18 + 3) && v6 == *(_QWORD *)(v32 + 8) )
        return 0;
LABEL_33:
      if ( !v32 )
        return 0;
      v5 = *(_QWORD *)(v32 + 8);
      if ( (v5 & 6) == 0 )
        return v32;
    }
    v20 = v12 + 1;
    if ( (unsigned int)v13 < v12 + 1 )
    {
      v27 = v20;
      if ( v20 >= v13 )
      {
        if ( v20 > v13 )
        {
          if ( v20 > (unsigned __int64)*(unsigned int *)(v11 + 412) )
          {
            sub_16CD150(v11 + 400, (const void *)(v11 + 416), v20, 8, a5, v14);
            v13 = *(unsigned int *)(v11 + 408);
            v14 = v10 & 0x7FFFFFFF;
            v15 = 8 * v14;
            v27 = v20;
          }
          v21 = *(_QWORD *)(v11 + 400);
          v28 = (_QWORD *)(v21 + 8 * v27);
          v29 = (_QWORD *)(v21 + 8 * v13);
          v30 = *(_QWORD *)(v11 + 416);
          if ( v28 != v29 )
          {
            do
              *v29++ = v30;
            while ( v28 != v29 );
            v21 = *(_QWORD *)(v11 + 400);
          }
          *(_DWORD *)(v11 + 408) = v20;
          goto LABEL_17;
        }
      }
      else
      {
        *(_DWORD *)(v11 + 408) = v20;
      }
    }
    v21 = *(_QWORD *)(v11 + 400);
LABEL_17:
    v31 = v14;
    *(_QWORD *)(v21 + v15) = sub_1DBA290(v10);
    v16 = *(_QWORD *)(*(_QWORD *)(v11 + 400) + 8 * v31);
    sub_1DBB110((_QWORD *)v11, v16);
    goto LABEL_10;
  }
  return v32;
}

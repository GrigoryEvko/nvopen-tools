// Function: sub_37C82C0
// Address: 0x37c82c0
//
bool __fastcall sub_37C82C0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  unsigned int *v7; // r11
  unsigned int v8; // ecx
  unsigned int v9; // edx
  unsigned int *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdx
  unsigned int *v13; // rsi
  __int64 v14; // r15
  __int64 v15; // r14
  __int64 v16; // r12
  __int64 v17; // rdx
  int v18; // eax
  __int64 *v20; // rdi
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 *v24; // rax
  __int64 v25; // r12
  __int64 *v26; // r13
  unsigned __int64 *v27; // r12
  unsigned __int64 *v28; // rsi
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rsi
  unsigned int *v31; // rsi
  __int64 v33; // [rsp+10h] [rbp-50h]
  __int64 v34; // [rsp+20h] [rbp-40h]
  unsigned __int64 *v35; // [rsp+28h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 432);
  v2 = *(_QWORD *)(v1 + 48);
  v33 = v2 + 96LL * *(unsigned int *)(v1 + 56);
  if ( v2 == v33 )
  {
    v18 = *(_DWORD *)(v1 + 56);
  }
  else
  {
    do
    {
      v3 = *(_QWORD *)(v2 + 16);
      v4 = 16LL * *(unsigned int *)(v2 + 24);
      v5 = v3 + v4;
      if ( v3 != v3 + v4 )
      {
        _BitScanReverse64(&v6, v4 >> 4);
        sub_37C80C0(*(_QWORD *)(v2 + 16), v3 + v4, 2LL * (int)(63 - (v6 ^ 0x3F)));
        if ( (unsigned __int64)v4 <= 0x100 )
        {
          sub_37B6600(v3, (unsigned int *)(v3 + v4));
        }
        else
        {
          sub_37B6600(v3, (unsigned int *)(v3 + 256));
          for ( ; (unsigned int *)v5 != v7; *((_QWORD *)v13 + 1) = v11 )
          {
            while ( 1 )
            {
              v8 = *v7;
              v9 = *(v7 - 4);
              v10 = v7 - 4;
              v11 = *((_QWORD *)v7 + 1);
              if ( *v7 < v9 )
                break;
              v31 = v7;
              v7 += 4;
              *v31 = v8;
              *((_QWORD *)v31 + 1) = v11;
              if ( (unsigned int *)v5 == v7 )
                goto LABEL_8;
            }
            do
            {
              v10[4] = v9;
              v12 = *((_QWORD *)v10 + 1);
              v13 = v10;
              v10 -= 4;
              *((_QWORD *)v10 + 5) = v12;
              v9 = *v10;
            }
            while ( v8 < *v10 );
            v7 += 4;
            *v13 = v8;
          }
        }
      }
LABEL_8:
      v14 = *(_QWORD *)(v2 + 8);
      if ( v14 )
      {
        v15 = *(_QWORD *)(v2 + 16);
        v16 = v15 + 16LL * *(unsigned int *)(v2 + 24);
        while ( v15 != v16 )
        {
          v17 = *(_QWORD *)(v15 + 8);
          v15 += 16;
          sub_2E326B0(v14, *(__int64 **)v2, v17);
        }
      }
      else
      {
        v20 = *(__int64 **)v2;
        v21 = *(_DWORD *)(*(_QWORD *)v2 + 44LL);
        if ( (v21 & 4) != 0 || (v21 & 8) == 0 )
          v22 = (*(_QWORD *)(v20[2] + 24) >> 9) & 1LL;
        else
          LOBYTE(v22) = sub_2E88A90((__int64)v20, 512, 1);
        if ( !(_BYTE)v22 )
        {
          v23 = *(_QWORD *)(v2 + 16);
          v24 = *(__int64 **)v2;
          v25 = 16LL * *(unsigned int *)(v2 + 24);
          v26 = *(__int64 **)(*(_QWORD *)v2 + 24LL);
          v34 = v23 + v25;
          if ( v23 + v25 != v23 )
          {
            while ( 1 )
            {
              for ( ; (*((_BYTE *)v24 + 44) & 8) != 0; v24 = (__int64 *)v24[1] )
                ;
              v27 = (unsigned __int64 *)(v26 + 6 == (__int64 *)(v26[6] & 0xFFFFFFFFFFFFFFF8LL) ? v26[7] : v24[1]);
              v28 = *(unsigned __int64 **)(v23 + 8);
              v35 = v28;
              v23 += 16;
              sub_2E31040(v26 + 5, (__int64)v28);
              v29 = *v28;
              v30 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
              v35[1] = (unsigned __int64)v27;
              *v35 = v30 | v29 & 7;
              *(_QWORD *)(v30 + 8) = v35;
              *v27 = *v27 & 7 | (unsigned __int64)v35;
              if ( v23 == v34 )
                break;
              v24 = *(__int64 **)v2;
            }
          }
        }
      }
      v2 += 96;
    }
    while ( v33 != v2 );
    v18 = *(_DWORD *)(*(_QWORD *)(a1 + 432) + 56LL);
  }
  return v18 != 0;
}

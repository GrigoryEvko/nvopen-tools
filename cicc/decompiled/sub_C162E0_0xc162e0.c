// Function: sub_C162E0
// Address: 0xc162e0
//
__int64 *__fastcall sub_C162E0(__int64 a1, __int64 a2, int *a3)
{
  __int64 *result; // rax
  __int64 v5; // r10
  __int64 v6; // rcx
  __int64 v7; // rdx
  int v8; // edx
  int v9; // edx
  int v10; // edx
  __int64 v11; // rdx
  int v12; // edx
  int v13; // edx
  __int64 v14; // rdx
  __int64 v15; // rdx
  int v16; // edx
  int v17; // edx
  int v18; // edx
  int v19; // edx
  int v20; // edx
  int v21; // edx
  int v22; // edx
  int v23; // edx
  __int64 v24; // rdx
  int v25; // edx
  int v26; // edx
  int v27; // edx
  int v28; // edx
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  int v33; // edx

  *(_QWORD *)a1 = 0;
  result = *(__int64 **)a2;
  v5 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( v5 != *(_QWORD *)a2 )
  {
    do
    {
      v6 = *result;
      switch ( *result )
      {
        case 1LL:
          v33 = *a3++;
          *(_DWORD *)(a1 + 8) = v33;
          break;
        case 2LL:
          v32 = *(_QWORD *)a3;
          a3 += 2;
          *(_QWORD *)(a1 + 16) = v32;
          break;
        case 3LL:
          v31 = *(_QWORD *)a3;
          a3 += 2;
          *(_QWORD *)(a1 + 24) = v31;
          break;
        case 4LL:
          v30 = *(_QWORD *)a3;
          a3 += 2;
          *(_QWORD *)(a1 + 32) = v30;
          break;
        case 5LL:
          v29 = *(_QWORD *)a3;
          a3 += 2;
          *(_QWORD *)(a1 + 40) = v29;
          break;
        case 6LL:
          v28 = *a3++;
          *(_DWORD *)(a1 + 48) = v28;
          break;
        case 7LL:
          v27 = *a3++;
          *(_DWORD *)(a1 + 52) = v27;
          break;
        case 8LL:
          v26 = *a3++;
          *(_DWORD *)(a1 + 56) = v26;
          break;
        case 9LL:
          v25 = *a3++;
          *(_DWORD *)(a1 + 60) = v25;
          break;
        case 10LL:
          v24 = *(_QWORD *)a3;
          a3 += 2;
          *(_QWORD *)(a1 + 64) = v24;
          break;
        case 11LL:
          v23 = *a3++;
          *(_DWORD *)(a1 + 72) = v23;
          break;
        case 12LL:
          v22 = *a3++;
          *(_DWORD *)(a1 + 76) = v22;
          break;
        case 13LL:
          v21 = *a3++;
          *(_DWORD *)(a1 + 80) = v21;
          break;
        case 14LL:
          v20 = *a3++;
          *(_DWORD *)(a1 + 84) = v20;
          break;
        case 15LL:
          v19 = *a3++;
          *(_DWORD *)(a1 + 88) = v19;
          break;
        case 16LL:
          v18 = *a3++;
          *(_DWORD *)(a1 + 92) = v18;
          break;
        case 17LL:
          v17 = *a3++;
          *(_DWORD *)(a1 + 96) = v17;
          break;
        case 18LL:
          v16 = *a3++;
          *(_DWORD *)(a1 + 100) = v16;
          break;
        case 19LL:
          v15 = *(_QWORD *)a3;
          a3 += 2;
          *(_QWORD *)(a1 + 104) = v15;
          break;
        case 20LL:
          v14 = *(_QWORD *)a3;
          a3 += 2;
          *(_QWORD *)(a1 + 112) = v14;
          break;
        case 21LL:
          v13 = *a3++;
          *(_DWORD *)(a1 + 120) = v13;
          break;
        case 22LL:
          v12 = *a3++;
          *(_DWORD *)(a1 + 124) = v12;
          break;
        case 23LL:
          v11 = *(_QWORD *)a3;
          a3 += 2;
          *(_QWORD *)(a1 + 128) = v11;
          break;
        case 24LL:
          v10 = *a3++;
          *(_DWORD *)(a1 + 136) = v10;
          break;
        case 25LL:
          v9 = *a3++;
          *(_DWORD *)(a1 + 140) = v9;
          break;
        case 26LL:
          v8 = *a3++;
          *(_DWORD *)(a1 + 144) = v8;
          break;
        case 27LL:
          v7 = *(_QWORD *)a3;
          a3 += 2;
          *(_QWORD *)(a1 + 152) = v7;
          break;
        default:
          BUG();
      }
      ++result;
      *(_QWORD *)a1 |= 1LL << v6;
    }
    while ( (__int64 *)v5 != result );
  }
  return result;
}

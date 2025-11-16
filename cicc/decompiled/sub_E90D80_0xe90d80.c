// Function: sub_E90D80
// Address: 0xe90d80
//
unsigned __int64 __fastcall sub_E90D80(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // rbx
  unsigned __int64 v7; // r15
  int v8; // esi
  unsigned __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned __int64 v11; // rbx
  int v12; // ecx
  int v13; // edx
  __int64 v14; // rsi
  __int64 v15; // rdx
  unsigned __int64 v16; // rdx
  int v18; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h]

  result = a2 - a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(a2 - a1) >> 3);
  if ( (__int64)(a2 - a1) > 24 )
  {
    v6 = (v5 - 2) / 2;
    v7 = a1 + 8 * (v6 + ((v5 - 2 + ((unsigned __int64)(v5 - 2) >> 63)) & 0xFFFFFFFFFFFFFFFELL));
    while ( 1 )
    {
      v8 = *(_DWORD *)v7;
      v9 = *(_QWORD *)(v7 + 8);
      v7 -= 24LL;
      v10 = *(_QWORD *)(v7 + 40);
      v18 = v8;
      v19 = v9;
      v20 = v10;
      result = sub_E8F100(a1, v6, v5, &v18);
      if ( !v6 )
        break;
      --v6;
    }
  }
  v11 = a2;
  if ( a3 > a2 )
  {
    while ( 1 )
    {
      result = *(_QWORD *)(v11 + 8);
      v16 = *(_QWORD *)(a1 + 8);
      if ( result < v16 )
        break;
      if ( result != v16 )
        goto LABEL_9;
      v12 = *(_DWORD *)v11;
      if ( *(_DWORD *)v11 < *(_DWORD *)a1 )
      {
LABEL_8:
        *(_QWORD *)(v11 + 8) = v16;
        v13 = *(_DWORD *)a1;
        v14 = *(_QWORD *)(v11 + 16);
        v18 = v12;
        *(_DWORD *)v11 = v13;
        v15 = *(_QWORD *)(a1 + 16);
        v20 = v14;
        *(_QWORD *)(v11 + 16) = v15;
        v19 = result;
        result = sub_E8F100(a1, 0, v5, &v18);
LABEL_9:
        v11 += 24LL;
        if ( a3 <= v11 )
          return result;
      }
      else
      {
        v11 += 24LL;
        if ( a3 <= v11 )
          return result;
      }
    }
    v12 = *(_DWORD *)v11;
    goto LABEL_8;
  }
  return result;
}

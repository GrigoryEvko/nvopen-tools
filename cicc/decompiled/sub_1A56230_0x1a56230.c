// Function: sub_1A56230
// Address: 0x1a56230
//
unsigned __int64 __fastcall sub_1A56230(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 *v6; // r13
  unsigned __int64 v7; // rcx
  __int64 v8; // rdi
  unsigned int v9; // esi
  unsigned int v10; // r9d
  __int64 v11; // r8
  unsigned int v12; // edx
  __int64 v13; // r10
  __int64 v14; // r12
  __int64 v15; // rsi
  _BYTE *v16; // rsi
  __int64 v17; // r8
  __int64 *v18; // rdi
  unsigned int v19; // r9d
  _QWORD *v20; // rsi
  int v21; // r10d
  unsigned __int64 v22; // r12
  unsigned int v23; // r11d
  __int64 v24; // rdx
  unsigned int v25; // r11d
  int v26; // ecx
  int v27; // edx
  __int64 v28; // rdx
  __int64 v29; // [rsp+0h] [rbp-90h]
  __int64 *i; // [rsp+18h] [rbp-78h]
  unsigned __int64 v32; // [rsp+28h] [rbp-68h]
  unsigned __int64 v33; // [rsp+28h] [rbp-68h]
  unsigned __int64 v34; // [rsp+28h] [rbp-68h]
  __int64 v35; // [rsp+38h] [rbp-58h] BYREF
  unsigned __int64 v36[2]; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v37; // [rsp+50h] [rbp-40h]

  v29 = a3 + 32;
  sub_13FC0C0(a3 + 32, (unsigned int)((__int64)(*(_QWORD *)(a2 + 40) - *(_QWORD *)(a2 + 32)) >> 3));
  result = *(_QWORD *)(a2 + 40);
  v6 = *(__int64 **)(a2 + 32);
  for ( i = (__int64 *)result; i != v6; ++v6 )
  {
    v14 = *v6;
    v15 = *a1;
    v35 = *v6;
    sub_1A51850(v36, v15, &v35);
    v7 = v37;
    if ( v37 != -8 && v37 != 0 && v37 != -16 )
    {
      v33 = v37;
      sub_1649B30(v36);
      v7 = v33;
    }
    v36[0] = v7;
    v16 = *(_BYTE **)(a3 + 40);
    if ( v16 == *(_BYTE **)(a3 + 48) )
    {
      v34 = v7;
      sub_1292090(v29, v16, v36);
      v17 = v36[0];
      v7 = v34;
    }
    else
    {
      if ( v16 )
      {
        *(_QWORD *)v16 = v7;
        v16 = *(_BYTE **)(a3 + 40);
      }
      v17 = v7;
      *(_QWORD *)(a3 + 40) = v16 + 8;
    }
    result = *(_QWORD *)(a3 + 64);
    if ( *(_QWORD *)(a3 + 72) != result )
      goto LABEL_3;
    v18 = (__int64 *)(result + 8LL * *(unsigned int *)(a3 + 84));
    v19 = *(_DWORD *)(a3 + 84);
    if ( (__int64 *)result != v18 )
    {
      v20 = 0;
      while ( v17 != *(_QWORD *)result )
      {
        if ( *(_QWORD *)result == -2 )
          v20 = (_QWORD *)result;
        result += 8LL;
        if ( v18 == (__int64 *)result )
        {
          if ( !v20 )
            goto LABEL_28;
          *v20 = v17;
          --*(_DWORD *)(a3 + 88);
          ++*(_QWORD *)(a3 + 56);
          goto LABEL_4;
        }
      }
      goto LABEL_4;
    }
LABEL_28:
    if ( v19 < *(_DWORD *)(a3 + 80) )
    {
      *(_DWORD *)(a3 + 84) = v19 + 1;
      *v18 = v17;
      ++*(_QWORD *)(a3 + 56);
    }
    else
    {
LABEL_3:
      v32 = v7;
      result = (unsigned __int64)sub_16CCBA0(a3 + 56, v17);
      v7 = v32;
    }
LABEL_4:
    v8 = a1[1];
    v9 = *(_DWORD *)(v8 + 24);
    if ( v9 )
    {
      v10 = v9 - 1;
      v11 = *(_QWORD *)(v8 + 8);
      v12 = (v9 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      result = v11 + 16LL * v12;
      v13 = *(_QWORD *)result;
      if ( v14 != *(_QWORD *)result )
      {
        result = 1;
        while ( v13 != -8 )
        {
          v25 = result + 1;
          v12 = v10 & (result + v12);
          result = v11 + 16LL * v12;
          v13 = *(_QWORD *)result;
          if ( v14 == *(_QWORD *)result )
            goto LABEL_6;
          result = v25;
        }
        continue;
      }
LABEL_6:
      if ( a2 == *(_QWORD *)(result + 8) )
      {
        v35 = v7;
        v21 = 1;
        v22 = 0;
        v23 = v10 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        result = v11 + 16LL * v23;
        v24 = *(_QWORD *)result;
        if ( v7 == *(_QWORD *)result )
        {
LABEL_27:
          *(_QWORD *)(result + 8) = a3;
          continue;
        }
        while ( v24 != -8 )
        {
          if ( v24 == -16 && !v22 )
            v22 = result;
          v23 = v10 & (v21 + v23);
          result = v11 + 16LL * v23;
          v24 = *(_QWORD *)result;
          if ( v7 == *(_QWORD *)result )
            goto LABEL_27;
          ++v21;
        }
        v26 = *(_DWORD *)(v8 + 16);
        if ( v22 )
          result = v22;
        ++*(_QWORD *)v8;
        v27 = v26 + 1;
        if ( 4 * (v26 + 1) >= 3 * v9 )
        {
          v9 *= 2;
        }
        else if ( v9 - *(_DWORD *)(v8 + 20) - v27 > v9 >> 3 )
        {
LABEL_44:
          *(_DWORD *)(v8 + 16) = v27;
          if ( *(_QWORD *)result != -8 )
            --*(_DWORD *)(v8 + 20);
          v28 = v35;
          *(_QWORD *)(result + 8) = 0;
          *(_QWORD *)result = v28;
          goto LABEL_27;
        }
        sub_1400170(v8, v9);
        sub_13FD8B0(v8, &v35, v36);
        result = v36[0];
        v27 = *(_DWORD *)(v8 + 16) + 1;
        goto LABEL_44;
      }
    }
  }
  return result;
}

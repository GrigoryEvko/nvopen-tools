// Function: sub_29024C0
// Address: 0x29024c0
//
_QWORD *__fastcall sub_29024C0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rax
  unsigned int v8; // esi
  __int64 v9; // rdi
  int v10; // r15d
  __int64 *v11; // r9
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // r11
  __int64 v15; // r15
  __int64 v16; // r14
  int v17; // ecx
  int v18; // ecx
  __int64 v19; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v20[7]; // [rsp+18h] [rbp-38h] BYREF

  result = v20;
  if ( a1 )
  {
    v5 = a1;
    while ( 1 )
    {
      v6 = *(_QWORD *)(v5 + 24);
      if ( *(_BYTE *)v6 != 85 )
        goto LABEL_3;
      result = *(_QWORD **)(v6 - 32);
      if ( !result
        || *(_BYTE *)result
        || result[3] != *(_QWORD *)(v6 + 80)
        || (*((_BYTE *)result + 33) & 0x20) == 0
        || *((_DWORD *)result + 9) != 149 )
      {
        goto LABEL_3;
      }
      v7 = sub_B5B890(*(_QWORD *)(v5 + 24));
      v8 = *(_DWORD *)(a3 + 24);
      v19 = v7;
      if ( !v8 )
        break;
      v9 = *(_QWORD *)(a3 + 8);
      v10 = 1;
      v11 = 0;
      v12 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v13 = (__int64 *)(v9 + 16LL * v12);
      v14 = *v13;
      if ( v7 != *v13 )
      {
        while ( v14 != -4096 )
        {
          if ( v14 == -8192 && !v11 )
            v11 = v13;
          v12 = (v8 - 1) & (v10 + v12);
          v13 = (__int64 *)(v9 + 16LL * v12);
          v14 = *v13;
          if ( v7 == *v13 )
            goto LABEL_12;
          ++v10;
        }
        v17 = *(_DWORD *)(a3 + 16);
        if ( !v11 )
          v11 = v13;
        ++*(_QWORD *)a3;
        v18 = v17 + 1;
        v20[0] = v11;
        if ( 4 * v18 < 3 * v8 )
        {
          if ( v8 - *(_DWORD *)(a3 + 20) - v18 > v8 >> 3 )
          {
LABEL_26:
            *(_DWORD *)(a3 + 16) = v18;
            if ( *v11 != -4096 )
              --*(_DWORD *)(a3 + 20);
            *v11 = v7;
            v15 = 0;
            v11[1] = 0;
            goto LABEL_13;
          }
LABEL_31:
          sub_29022E0(a3, v8);
          sub_2901330(a3, &v19, v20);
          v7 = v19;
          v11 = (__int64 *)v20[0];
          v18 = *(_DWORD *)(a3 + 16) + 1;
          goto LABEL_26;
        }
LABEL_30:
        v8 *= 2;
        goto LABEL_31;
      }
LABEL_12:
      v15 = v13[1];
LABEL_13:
      v16 = *(_QWORD *)(v6 + 32);
      result = sub_BD2C40(80, unk_3F10A10);
      if ( result )
      {
        result = (_QWORD *)sub_B4D460((__int64)result, v6, v15, v16, 0);
        v5 = *(_QWORD *)(v5 + 8);
        if ( !v5 )
          return result;
      }
      else
      {
LABEL_3:
        v5 = *(_QWORD *)(v5 + 8);
        if ( !v5 )
          return result;
      }
    }
    ++*(_QWORD *)a3;
    v20[0] = 0;
    goto LABEL_30;
  }
  return result;
}

// Function: sub_29638A0
// Address: 0x29638a0
//
__int64 __fastcall sub_29638A0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rcx
  __int64 *v9; // r15
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // esi
  unsigned __int64 v15; // r12
  _BYTE *v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rdi
  unsigned int v20; // esi
  unsigned int v21; // edx
  __int64 v22; // r10
  unsigned __int64 v23; // r13
  unsigned int v24; // r11d
  __int64 *v25; // rax
  __int64 v26; // rdx
  int v27; // eax
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // [rsp+8h] [rbp-88h]
  __int64 *v32; // [rsp+28h] [rbp-68h]
  unsigned __int64 v33; // [rsp+38h] [rbp-58h] BYREF
  unsigned __int64 v34[2]; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v35; // [rsp+50h] [rbp-40h]

  v30 = a3 + 32;
  result = sub_D4ACD0(a3 + 32, (unsigned int)((__int64)(*(_QWORD *)(a2 + 40) - *(_QWORD *)(a2 + 32)) >> 3));
  v8 = *(_QWORD *)(a2 + 40);
  v32 = (__int64 *)v8;
  if ( v8 != *(_QWORD *)(a2 + 32) )
  {
    v9 = *(__int64 **)(a2 + 32);
    while ( 1 )
    {
      v10 = *a1;
      v11 = *v9;
      v12 = *(unsigned int *)(*a1 + 24);
      if ( !(_DWORD)v12 )
        goto LABEL_28;
      v13 = *(_QWORD *)(v10 + 8);
      v7 = (unsigned int)(v12 - 1);
      v14 = v7 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v10 = v13 + ((unsigned __int64)v14 << 6);
      v6 = *(_QWORD *)(v10 + 24);
      if ( v11 != v6 )
        break;
LABEL_5:
      if ( v10 == v13 + (v12 << 6) )
        goto LABEL_28;
      v34[0] = 6;
      v34[1] = 0;
      v35 = *(_QWORD *)(v10 + 56);
      v15 = v35;
      if ( v35 != -4096 && v35 != 0 && v35 != -8192 )
      {
        sub_BD6050(v34, *(_QWORD *)(v10 + 40) & 0xFFFFFFFFFFFFFFF8LL);
        v15 = v35;
        LOBYTE(v10) = v35 != -4096;
        if ( ((v35 != -8192) & (unsigned __int8)v10) != 0 )
        {
          if ( v35 )
            sub_BD60C0(v34);
        }
      }
      v34[0] = v15;
      v16 = *(_BYTE **)(a3 + 40);
      if ( v16 != *(_BYTE **)(a3 + 48) )
      {
LABEL_12:
        if ( v16 )
        {
          *(_QWORD *)v16 = v15;
          v16 = *(_BYTE **)(a3 + 40);
        }
        *(_QWORD *)(a3 + 40) = v16 + 8;
        v17 = v15;
        goto LABEL_15;
      }
LABEL_29:
      sub_9319A0(v30, v16, v34);
      v17 = v34[0];
LABEL_15:
      if ( !*(_BYTE *)(a3 + 84) )
        goto LABEL_25;
      result = *(_QWORD *)(a3 + 64);
      v18 = *(unsigned int *)(a3 + 76);
      v10 = result + 8 * v18;
      if ( result != v10 )
      {
        while ( *(_QWORD *)result != v17 )
        {
          result += 8;
          if ( v10 == result )
            goto LABEL_30;
        }
        goto LABEL_20;
      }
LABEL_30:
      if ( (unsigned int)v18 < *(_DWORD *)(a3 + 72) )
      {
        *(_DWORD *)(a3 + 76) = v18 + 1;
        *(_QWORD *)v10 = v17;
        ++*(_QWORD *)(a3 + 56);
      }
      else
      {
LABEL_25:
        result = (__int64)sub_C8CC70(a3 + 56, v17, v10, v8, v6, v7);
      }
LABEL_20:
      v19 = a1[1];
      v20 = *(_DWORD *)(v19 + 24);
      v6 = *(_QWORD *)(v19 + 8);
      if ( v20 )
      {
        v7 = v20 - 1;
        v21 = v7 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        result = v6 + 16LL * v21;
        v22 = *(_QWORD *)result;
        if ( v11 != *(_QWORD *)result )
        {
          result = 1;
          while ( v22 != -4096 )
          {
            v8 = (unsigned int)(result + 1);
            v21 = v7 & (result + v21);
            result = v6 + 16LL * v21;
            v22 = *(_QWORD *)result;
            if ( v11 == *(_QWORD *)result )
              goto LABEL_22;
            result = (unsigned int)v8;
          }
          goto LABEL_23;
        }
LABEL_22:
        v8 = a2;
        if ( a2 == *(_QWORD *)(result + 8) )
        {
          v33 = v15;
          v8 = 1;
          v23 = 0;
          v24 = v7 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v25 = (__int64 *)(v6 + 16LL * v24);
          v26 = *v25;
          if ( *v25 == v15 )
          {
LABEL_33:
            result = (__int64)(v25 + 1);
            *(_QWORD *)result = a3;
            goto LABEL_23;
          }
          while ( v26 != -4096 )
          {
            if ( v26 == -8192 && !v23 )
              v23 = (unsigned __int64)v25;
            v24 = v7 & (v8 + v24);
            v25 = (__int64 *)(v6 + 16LL * v24);
            v26 = *v25;
            if ( *v25 == v15 )
              goto LABEL_33;
            v8 = (unsigned int)(v8 + 1);
          }
          if ( !v23 )
            v23 = (unsigned __int64)v25;
          v34[0] = v23;
          v27 = *(_DWORD *)(v19 + 16);
          ++*(_QWORD *)v19;
          v28 = v27 + 1;
          v7 = (unsigned int)(4 * v28);
          if ( (unsigned int)v7 >= 3 * v20 )
          {
            v20 *= 2;
          }
          else
          {
            v6 = v20 >> 3;
            if ( v20 - *(_DWORD *)(v19 + 20) - v28 > (unsigned int)v6 )
            {
LABEL_50:
              *(_DWORD *)(v19 + 16) = v28;
              v25 = (__int64 *)v34[0];
              if ( *(_QWORD *)v34[0] != -4096 )
                --*(_DWORD *)(v19 + 20);
              v29 = v33;
              v25[1] = 0;
              *v25 = v29;
              goto LABEL_33;
            }
          }
          sub_D4F150(v19, v20);
          sub_D4C730(v19, (__int64 *)&v33, v34);
          v28 = *(_DWORD *)(v19 + 16) + 1;
          goto LABEL_50;
        }
      }
LABEL_23:
      if ( v32 == ++v9 )
        return result;
    }
    v10 = 1;
    while ( v6 != -4096 )
    {
      v8 = (unsigned int)(v10 + 1);
      v14 = v7 & (v10 + v14);
      v10 = v13 + ((unsigned __int64)v14 << 6);
      v6 = *(_QWORD *)(v10 + 24);
      if ( v11 == v6 )
        goto LABEL_5;
      v10 = (unsigned int)v8;
    }
LABEL_28:
    v15 = 0;
    v16 = *(_BYTE **)(a3 + 40);
    v34[0] = 0;
    if ( v16 != *(_BYTE **)(a3 + 48) )
      goto LABEL_12;
    goto LABEL_29;
  }
  return result;
}

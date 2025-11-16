// Function: sub_26DC0A0
// Address: 0x26dc0a0
//
__int64 __fastcall sub_26DC0A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r13
  __int64 v4; // rbx
  char v5; // r15
  __int64 v7; // r12
  unsigned int v8; // esi
  __int64 v9; // r15
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 *v12; // rdi
  __int64 v13; // rdx
  __int64 *v14; // rax
  __int64 v15; // r8
  _QWORD *v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 *v25; // rsi
  __int64 v26; // r13
  __int64 *v27; // rax
  int v28; // eax
  unsigned __int8 v31; // [rsp+27h] [rbp-69h]
  __int64 v32; // [rsp+28h] [rbp-68h]
  __int64 v33; // [rsp+30h] [rbp-60h] BYREF
  __int64 *v34; // [rsp+38h] [rbp-58h] BYREF
  _QWORD v35[2]; // [rsp+40h] [rbp-50h] BYREF
  char v36; // [rsp+50h] [rbp-40h]

  v3 = (__int64 *)a1;
  v4 = *(_QWORD *)(a2 + 80);
  v32 = a2 + 72;
  v31 = *(_DWORD *)(a3 + 16) != 0;
  if ( v4 != a2 + 72 )
  {
    v5 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v7 = v4 - 24;
        if ( !v4 )
          v7 = 0;
        sub_26C6CF0((__int64)v35, (void (__fastcall ***)(unsigned __int64 *, _QWORD, __int64))a1, v7);
        if ( (v36 & 1) == 0 )
          break;
        v4 = *(_QWORD *)(v4 + 8);
        if ( v32 == v4 )
          goto LABEL_17;
      }
      v8 = *(_DWORD *)(a1 + 64);
      v9 = v35[0];
      v33 = v7;
      if ( !v8 )
        break;
      v10 = *(_QWORD *)(a1 + 48);
      v11 = 1;
      v12 = 0;
      v13 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v14 = (__int64 *)(v10 + 16 * v13);
      v15 = *v14;
      if ( v7 != *v14 )
      {
        while ( v15 != -4096 )
        {
          if ( !v12 && v15 == -8192 )
            v12 = v14;
          v13 = (v8 - 1) & ((_DWORD)v11 + (_DWORD)v13);
          v14 = (__int64 *)(v10 + 16LL * (unsigned int)v13);
          v15 = *v14;
          if ( v7 == *v14 )
            goto LABEL_9;
          v11 = (unsigned int)(v11 + 1);
        }
        if ( !v12 )
          v12 = v14;
        v28 = *(_DWORD *)(a1 + 56);
        ++*(_QWORD *)(a1 + 40);
        v13 = (unsigned int)(v28 + 1);
        v34 = v12;
        if ( 4 * (int)v13 < 3 * v8 )
        {
          v15 = v7;
          v10 = v8 >> 3;
          if ( v8 - *(_DWORD *)(a1 + 60) - (unsigned int)v13 > (unsigned int)v10 )
          {
LABEL_40:
            *(_DWORD *)(a1 + 56) = v13;
            if ( *v12 != -4096 )
              --*(_DWORD *)(a1 + 60);
            *v12 = v15;
            v16 = v12 + 1;
            v12[1] = 0;
            goto LABEL_10;
          }
LABEL_45:
          sub_FE19E0(a1 + 40, v8);
          sub_26C35D0(a1 + 40, &v33, &v34);
          v15 = v33;
          v12 = v34;
          v13 = (unsigned int)(*(_DWORD *)(a1 + 56) + 1);
          goto LABEL_40;
        }
LABEL_44:
        v8 *= 2;
        goto LABEL_45;
      }
LABEL_9:
      v16 = v14 + 1;
LABEL_10:
      *v16 = v9;
      if ( !*(_BYTE *)(a1 + 132) )
        goto LABEL_21;
      v17 = *(_QWORD **)(a1 + 112);
      v18 = *(unsigned int *)(a1 + 124);
      v13 = (__int64)&v17[v18];
      if ( v17 != (_QWORD *)v13 )
      {
        while ( v7 != *v17 )
        {
          if ( (_QWORD *)v13 == ++v17 )
            goto LABEL_20;
        }
        v5 = 1;
        goto LABEL_16;
      }
LABEL_20:
      if ( (unsigned int)v18 < *(_DWORD *)(a1 + 120) )
      {
        v5 = 1;
        *(_DWORD *)(a1 + 124) = v18 + 1;
        *(_QWORD *)v13 = v7;
        ++*(_QWORD *)(a1 + 104);
      }
      else
      {
LABEL_21:
        v5 = 1;
        sub_C8CC70(a1 + 104, v7, v13, v11, v15, v10);
      }
LABEL_16:
      v4 = *(_QWORD *)(v4 + 8);
      if ( v32 == v4 )
      {
LABEL_17:
        v31 |= v5;
        v3 = (__int64 *)a1;
        goto LABEL_18;
      }
    }
    ++*(_QWORD *)(a1 + 40);
    v34 = 0;
    goto LABEL_44;
  }
LABEL_18:
  if ( v31 )
  {
    sub_B2F4C0(a2, *(_QWORD *)(v3[150] + 64) + 1LL, 0, (_BYTE *)a3);
    if ( !LOBYTE(qword_500BA28[8]) )
    {
      sub_26C14C0(v3, a2);
      sub_26D7890((__int64)v3, a2);
    }
    sub_26CA2E0((__int64)v3, a2);
    sub_26DBC20((__int64)v3, a2, v20, v21, v22, v23);
    if ( LOBYTE(qword_500BA28[8]) )
    {
      v24 = *(_QWORD *)(a2 + 80);
      v25 = v3;
      if ( v24 )
        v24 -= 24;
      v26 = (__int64)(v3 + 5);
      v34 = (__int64 *)v24;
      sub_26C6CF0((__int64)v35, (void (__fastcall ***)(unsigned __int64 *, _QWORD, __int64))v25, v24);
      if ( *sub_26CC460(v26, (__int64 *)&v34) )
      {
        v27 = sub_26CC460(v26, (__int64 *)&v34);
        sub_B2F4C0(a2, *v27, 0, (_BYTE *)a3);
      }
    }
  }
  return v31;
}

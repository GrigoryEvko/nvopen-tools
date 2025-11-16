// Function: sub_146A0D0
// Address: 0x146a0d0
//
__int64 __fastcall sub_146A0D0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 *v4; // r12
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 v8; // r15
  __int64 v9; // r13
  unsigned __int8 v10; // al
  __int64 v11; // rsi
  __int64 v12; // r9
  int v13; // edx
  int v14; // edx
  __int64 v15; // rsi
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // r10
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // rdi
  unsigned int v24; // r11d
  __int64 *v25; // rax
  __int64 v26; // rcx
  int v28; // eax
  int v29; // edi
  int v30; // r13d
  __int64 *v31; // r8
  int v32; // edi
  int v33; // edi
  int v34; // eax
  __int64 v35; // [rsp+0h] [rbp-60h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+20h] [rbp-40h] BYREF
  __int64 *v40; // [rsp+28h] [rbp-38h] BYREF

  if ( a4 > dword_4F9AD20 )
    return 0;
  v4 = (__int64 *)a1;
  v6 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v7 = *(__int64 **)(a1 - 8);
    v4 = &v7[v6];
  }
  else
  {
    v7 = (__int64 *)(a1 - v6 * 8);
  }
  v8 = 0;
  if ( v4 != v7 )
  {
    v38 = a2 + 56;
    do
    {
      v9 = *v7;
      v10 = *(_BYTE *)(*v7 + 16);
      if ( v10 <= 0x10u )
        goto LABEL_6;
      if ( v10 <= 0x17u )
        return 0;
      v11 = *(_QWORD *)(v9 + 40);
      v39 = *v7;
      if ( !sub_1377F70(v38, v11) )
        return 0;
      if ( *(_BYTE *)(v9 + 16) == 77 )
      {
        if ( **(_QWORD **)(a2 + 32) != *(_QWORD *)(v9 + 40) )
          return 0;
      }
      else if ( !(unsigned __int8)sub_1452C00(v9) )
      {
        return 0;
      }
      v12 = v39;
      if ( *(_BYTE *)(v39 + 16) != 77 )
      {
        v13 = *(_DWORD *)(a3 + 24);
        if ( !v13 )
          goto LABEL_19;
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a3 + 8);
        v16 = v14 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
        v17 = (__int64 *)(v15 + 16LL * v16);
        v18 = *v17;
        if ( v39 != *v17 )
        {
          v28 = 1;
          while ( v18 != -8 )
          {
            v29 = v28 + 1;
            v16 = v14 & (v28 + v16);
            v17 = (__int64 *)(v15 + 16LL * v16);
            v18 = *v17;
            if ( v39 == *v17 )
              goto LABEL_18;
            v28 = v29;
          }
LABEL_19:
          v20 = sub_146A0D0(v39, a2, a3, a4 + 1);
          v21 = *(_DWORD *)(a3 + 24);
          v12 = v20;
          if ( v21 )
          {
            v22 = v39;
            v23 = *(_QWORD *)(a3 + 8);
            v24 = (v21 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
            v25 = (__int64 *)(v23 + 16LL * v24);
            v26 = *v25;
            if ( *v25 == v39 )
              goto LABEL_21;
            v30 = 1;
            v31 = 0;
            while ( v26 != -8 )
            {
              if ( v26 == -16 && !v31 )
                v31 = v25;
              v34 = v30++;
              v24 = (v21 - 1) & (v34 + v24);
              v25 = (__int64 *)(v23 + 16LL * v24);
              v26 = *v25;
              if ( v39 == *v25 )
                goto LABEL_21;
            }
            v32 = *(_DWORD *)(a3 + 16);
            if ( v31 )
              v25 = v31;
            ++*(_QWORD *)a3;
            v33 = v32 + 1;
            if ( 4 * v33 < 3 * v21 )
            {
              if ( v21 - *(_DWORD *)(a3 + 20) - v33 > v21 >> 3 )
              {
LABEL_38:
                *(_DWORD *)(a3 + 16) = v33;
                if ( *v25 != -8 )
                  --*(_DWORD *)(a3 + 20);
                *v25 = v22;
                v25[1] = 0;
LABEL_21:
                v25[1] = v12;
                if ( !v12 )
                  return 0;
                goto LABEL_13;
              }
              v35 = v12;
LABEL_43:
              sub_1469F10(a3, v21);
              sub_1463B80(a3, &v39, &v40);
              v25 = v40;
              v22 = v39;
              v12 = v35;
              v33 = *(_DWORD *)(a3 + 16) + 1;
              goto LABEL_38;
            }
          }
          else
          {
            ++*(_QWORD *)a3;
          }
          v35 = v12;
          v21 *= 2;
          goto LABEL_43;
        }
LABEL_18:
        v19 = v17[1];
        if ( !v19 )
          goto LABEL_19;
        v12 = v19;
      }
LABEL_13:
      if ( v8 && v8 != v12 )
        return 0;
      v8 = v12;
LABEL_6:
      v7 += 3;
    }
    while ( v4 != v7 );
  }
  return v8;
}

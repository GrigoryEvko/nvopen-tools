// Function: sub_2BF3400
// Address: 0x2bf3400
//
__int64 __fastcall sub_2BF3400(__int64 a1, __int64 a2)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 *v9; // rbx
  _QWORD *v10; // rax
  __int64 v11; // r15
  __int64 v12; // rcx
  __int64 v13; // rbx
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 *v16; // rbx
  __int64 *v17; // rax
  __int64 v18; // rax
  int v19; // esi
  __int64 v20; // rdi
  __int64 v21; // r8
  int v22; // esi
  int v23; // edx
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // r11
  __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+18h] [rbp-68h]
  __int64 v30[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v31; // [rsp+40h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 104);
  if ( !*(_BYTE *)(a2 + 24) || *(_QWORD *)(*(_QWORD *)(a1 + 48) + 112LL) != a1 )
  {
    v5 = sub_2BF0570(a1);
    if ( *(_DWORD *)(v5 + 64) != 1 || (v6 = **(_QWORD **)(v5 + 56)) == 0 || *(_BYTE *)(v6 + 8) || !*(_BYTE *)(v6 + 128) )
    {
      v7 = sub_2BF0910(a1, a2);
      v8 = *(_QWORD *)(a2 + 904);
      *(_QWORD *)(v8 + 56) = v7 + 48;
      *(_QWORD *)(v8 + 48) = v7;
      *(_WORD *)(v8 + 64) = 0;
      v31 = 257;
      v9 = *(__int64 **)(a2 + 904);
      v10 = sub_BD2C40(72, unk_3F148B8);
      v11 = (__int64)v10;
      if ( v10 )
        sub_B4C8A0((__int64)v10, v9[9], 0, 0);
      (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v9[11] + 16LL))(
        v9[11],
        v11,
        v30,
        v9[7],
        v9[8]);
      v12 = *v9 + 16LL * *((unsigned int *)v9 + 2);
      v13 = *v9;
      v29 = v12;
      while ( v29 != v13 )
      {
        v14 = *(_QWORD *)(v13 + 8);
        v15 = *(_DWORD *)v13;
        v13 += 16;
        sub_B99FD0(v11, v15, v14);
      }
      v16 = *(__int64 **)(a2 + 928);
      if ( *(_DWORD *)(a1 + 88) == 1 )
      {
        v17 = *(__int64 **)(a1 + 80);
        if ( *v17 )
        {
          v28 = *v17;
          if ( sub_2BF0C10(*(_QWORD *)(a2 + 920), *v17) )
          {
            v18 = *(_QWORD *)(a2 + 896);
            v19 = *(_DWORD *)(v18 + 24);
            v20 = *(_QWORD *)(v28 + 128);
            v21 = *(_QWORD *)(v18 + 8);
            if ( !v19 )
            {
LABEL_20:
              sub_D5F1F0(*(_QWORD *)(a2 + 904), v11);
              *(_QWORD *)(a2 + 104) = v7;
              v30[0] = a1;
              *sub_2BF2B80(a2 + 120, v30) = v7;
              sub_2BF2D10((__int64 *)a1, a2);
              return sub_2BF0980(a1, a2);
            }
            v22 = v19 - 1;
            v23 = 1;
            v24 = v22 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
            v25 = (__int64 *)(v21 + 16LL * v24);
            v26 = *v25;
            if ( v20 != *v25 )
            {
              while ( v26 != -4096 )
              {
                v24 = v22 & (v23 + v24);
                v25 = (__int64 *)(v21 + 16LL * v24);
                v26 = *v25;
                if ( v20 == *v25 )
                  goto LABEL_17;
                ++v23;
              }
              goto LABEL_20;
            }
LABEL_17:
            v16 = (__int64 *)v25[1];
          }
        }
      }
      if ( v16 )
        sub_D4F330(v16, v7, *(_QWORD *)(a2 + 896));
      goto LABEL_20;
    }
  }
  v30[0] = a1;
  *sub_2BF2B80(a2 + 120, v30) = v4;
  return sub_2BF0980(a1, a2);
}

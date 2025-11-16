// Function: sub_2F4C3D0
// Address: 0x2f4c3d0
//
__int64 __fastcall sub_2F4C3D0(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  unsigned int v4; // r14d
  unsigned int v5; // r12d
  __int64 v6; // rax
  int v7; // r15d
  unsigned __int8 *v8; // rcx
  int v9; // edx
  int v10; // eax
  int v11; // ecx
  int v12; // r12d
  char v13; // r8
  int v14; // eax
  unsigned int v15; // eax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rax
  unsigned int v22; // r12d
  __int64 v23; // rdi
  __int64 v24; // rdx
  _DWORD *v25; // rax
  __int64 v26; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v27; // [rsp+18h] [rbp-38h]
  unsigned __int8 *v28; // [rsp+18h] [rbp-38h]

  v3 = sub_2E0B010(a2);
  v4 = *(_DWORD *)(a2 + 112);
  v5 = v3;
  v6 = v4 & 0x7FFFFFFF;
  v7 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 920LL) + 8 * v6);
  if ( v7 != 2 )
  {
    if ( v7 == 5 )
    {
      return (unsigned int)dword_5023988++;
    }
    else
    {
      v8 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 56LL) + 16 * v6) & 0xFFFFFFFFFFFFFFF8LL);
      if ( v8[41] )
        goto LABEL_7;
      if ( !*(_BYTE *)(a1 + 65) )
      {
        v23 = *(_QWORD *)(a1 + 48);
        v24 = v5 >> 4;
        v25 = (_DWORD *)(*(_QWORD *)v23 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v8 + 24LL));
        if ( *(_DWORD *)(v23 + 8) != *v25 )
        {
          v26 = *(_QWORD *)v23 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v8 + 24LL);
          v28 = v8;
          sub_2F60630(v23, v8, v24, v8);
          v25 = (_DWORD *)v26;
          LODWORD(v24) = v5 >> 4;
          v8 = v28;
        }
        if ( (unsigned int)v24 > 2 * v25[1] )
          goto LABEL_7;
      }
      if ( v7 == 1 && *(_DWORD *)(a2 + 8) && (v27 = v8, v17 = sub_2E13500(*(_QWORD *)(a1 + 16), a2), v8 = v27, v17) )
      {
        v18 = *(_QWORD *)(a1 + 56);
        v19 = *(__int64 **)a2;
        if ( *(_BYTE *)(a1 + 65) )
        {
          v22 = *(_DWORD *)((v19[3 * *(unsigned int *)(a2 + 8) - 2] & 0xFFFFFFFFFFFFFFF8LL) + 24)
              - *(_DWORD *)((*(_QWORD *)(v18 + 104) & 0xFFFFFFFFFFFFFFF8LL) + 24);
          v9 = 0;
        }
        else
        {
          v20 = *(_QWORD *)(v18 + 96);
          v21 = *v19;
          v9 = 0;
          v22 = *(_DWORD *)((v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) - *(_DWORD *)((v21 & 0xFFFFFFFFFFFFFFF8LL) + 24);
        }
        v5 = v22 >> 2;
      }
      else
      {
LABEL_7:
        v9 = 1;
      }
      v10 = 0xFFFFFF;
      v11 = v8[40];
      if ( v5 <= 0xFFFFFF )
        v10 = v5;
      if ( *(_BYTE *)(a1 + 64) )
        v12 = (v11 << 25) | v10 | (v9 << 24);
      else
        v12 = (v11 << 24) | v10 | (v9 << 29);
      v13 = sub_300C0A0(*(_QWORD *)(a1 + 24), v4);
      v14 = v12;
      v5 = v12 | 0xC0000000;
      v15 = v14 | 0x80000000;
      if ( !v13 )
        return v15;
    }
  }
  return v5;
}

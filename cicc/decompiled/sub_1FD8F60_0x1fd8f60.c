// Function: sub_1FD8F60
// Address: 0x1fd8f60
//
__int64 __fastcall sub_1FD8F60(_QWORD *a1, __int64 a2)
{
  unsigned int v3; // r13d
  _QWORD *v4; // r12
  unsigned int v5; // eax
  unsigned __int8 v6; // bl
  unsigned int v7; // r15d
  unsigned __int8 v8; // al
  __int64 v9; // r8
  __int64 v10; // rsi
  unsigned int v11; // r9d
  __int64 v12; // r13
  __int64 v13; // rcx
  unsigned int v14; // eax
  _QWORD *v15; // rbx
  _QWORD *v16; // rdx
  unsigned int v17; // eax
  __int64 v19; // rax
  int v20; // eax
  int v21; // edx
  __int64 v22; // rsi
  unsigned int v23; // eax
  _QWORD *v24; // rcx
  __int64 v25; // rcx
  __int64 v26; // r8
  int v27; // r9d
  int v28; // r11d
  _QWORD *v29; // r10
  int v30; // eax
  int v31; // esi
  int v32; // edi
  __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+8h] [rbp-68h]
  _QWORD *v35; // [rsp+10h] [rbp-60h] BYREF
  __int64 v36; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v37; // [rsp+20h] [rbp-50h] BYREF
  __int64 v38[9]; // [rsp+28h] [rbp-48h] BYREF

  v3 = 0;
  v4 = (_QWORD *)a2;
  LOBYTE(v5) = sub_1FD35E0(a1[12], *(_QWORD *)a2);
  if ( (_BYTE)v5 )
  {
    v6 = v5;
    v7 = v5;
    if ( !*(_QWORD *)(a1[14] + 8LL * (unsigned __int8)v5 + 120) )
    {
      v33 = a1[14];
      if ( (unsigned __int8)(v5 - 2) > 2u )
        return v3;
      v19 = sub_16498A0(a2);
      sub_1F40D10((__int64)&v37, v33, v19, v6, 0);
      v7 = LOBYTE(v38[0]);
    }
    v3 = sub_1FD4C00((__int64)a1, a2);
    if ( !v3 )
    {
      v8 = *(_BYTE *)(a2 + 16);
      if ( v8 <= 0x17u )
      {
LABEL_15:
        sub_1FD3BC0(&v35, (__int64)a1);
        v3 = sub_1FD8DB0((__int64)a1, (__int64)v4, v7, v25, v26, v27);
        v38[0] = v36;
        v37 = v35;
        if ( v36 )
          sub_1623A60((__int64)v38, v36, 2);
        sub_1FD3C80(a1, &v37);
        if ( v38[0] )
          sub_161E7C0((__int64)v38, v38[0]);
        if ( v36 )
          sub_161E7C0((__int64)&v36, v36);
        return v3;
      }
      v9 = a1[5];
      if ( v8 == 53 )
      {
        v20 = *(_DWORD *)(v9 + 360);
        if ( v20 )
        {
          v21 = v20 - 1;
          v22 = *(_QWORD *)(v9 + 344);
          v23 = (v20 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v24 = *(_QWORD **)(v22 + 16LL * v23);
          if ( v4 == v24 )
            goto LABEL_15;
          v32 = 1;
          while ( v24 != (_QWORD *)-8LL )
          {
            v23 = v21 & (v32 + v23);
            v24 = *(_QWORD **)(v22 + 16LL * v23);
            if ( v4 == v24 )
              goto LABEL_15;
            ++v32;
          }
        }
      }
      v10 = *v4;
      v35 = v4;
      if ( *(_BYTE *)(v10 + 8) == 10 )
        return v3;
      v11 = *(_DWORD *)(v9 + 232);
      v12 = v9 + 208;
      if ( v11 )
      {
        v13 = *(_QWORD *)(v9 + 216);
        v14 = (v11 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v15 = (_QWORD *)(v13 + 16LL * v14);
        v16 = (_QWORD *)*v15;
        if ( v4 == (_QWORD *)*v15 )
        {
LABEL_9:
          v17 = sub_1FDE000(v9, v10);
          *((_DWORD *)v15 + 2) = v17;
          return v17;
        }
        v28 = 1;
        v29 = 0;
        while ( v16 != (_QWORD *)-8LL )
        {
          if ( !v29 && v16 == (_QWORD *)-16LL )
            v29 = v15;
          v14 = (v11 - 1) & (v28 + v14);
          v15 = (_QWORD *)(v13 + 16LL * v14);
          v16 = (_QWORD *)*v15;
          if ( v4 == (_QWORD *)*v15 )
            goto LABEL_9;
          ++v28;
        }
        v30 = *(_DWORD *)(v9 + 224);
        if ( v29 )
          v15 = v29;
        ++*(_QWORD *)(v9 + 208);
        if ( 4 * (v30 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(v9 + 228) - (v30 + 1) > v11 >> 3 )
          {
LABEL_27:
            ++*(_DWORD *)(v9 + 224);
            if ( *v15 != -8 )
              --*(_DWORD *)(v9 + 228);
            *v15 = v4;
            *((_DWORD *)v15 + 2) = 0;
            v10 = *v35;
            goto LABEL_9;
          }
          v34 = v9;
          v31 = v11;
LABEL_32:
          sub_1542080(v9 + 208, v31);
          sub_154CC80(v12, (__int64 *)&v35, &v37);
          v15 = v37;
          v4 = v35;
          v9 = v34;
          goto LABEL_27;
        }
      }
      else
      {
        ++*(_QWORD *)(v9 + 208);
      }
      v34 = v9;
      v31 = 2 * v11;
      goto LABEL_32;
    }
  }
  return v3;
}

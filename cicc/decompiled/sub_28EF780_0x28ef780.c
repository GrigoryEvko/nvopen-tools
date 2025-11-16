// Function: sub_28EF780
// Address: 0x28ef780
//
__int64 __fastcall sub_28EF780(__int64 a1, _BYTE *a2)
{
  __int64 v3; // r13
  unsigned int *v4; // rbx
  _BYTE *v5; // rax
  unsigned int v6; // r15d
  __int64 *v7; // rdx
  unsigned int v8; // r10d
  __int64 v9; // rbx
  __int64 v10; // rcx
  __int64 *v11; // rdi
  unsigned int v12; // esi
  __int64 v13; // rax
  __int64 *v14; // r11
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rbx
  unsigned int v18; // r13d
  __int64 v19; // r15
  __int64 v20; // r14
  __int64 v21; // rsi
  unsigned int v22; // eax
  unsigned int v23; // r15d
  unsigned __int8 v24; // al
  unsigned int v26; // esi
  int v27; // eax
  __int64 v28; // rbx
  int v29; // eax
  _BYTE *v30; // r12
  int v31; // eax
  int v32; // esi
  __int64 v33; // [rsp+8h] [rbp-98h]
  bool v34; // [rsp+1Bh] [rbp-85h]
  unsigned int v35; // [rsp+1Ch] [rbp-84h]
  __int64 v36; // [rsp+20h] [rbp-80h] BYREF
  __int64 v37; // [rsp+28h] [rbp-78h] BYREF
  __int64 *v38[2]; // [rsp+30h] [rbp-70h] BYREF
  _BYTE *v39; // [rsp+40h] [rbp-60h]
  __int64 *v40; // [rsp+50h] [rbp-50h] BYREF
  __int64 v41; // [rsp+58h] [rbp-48h]
  __int64 v42; // [rsp+60h] [rbp-40h]

  v3 = (__int64)a2;
  if ( *a2 <= 0x1Cu )
  {
    v6 = 0;
    if ( *a2 == 22 )
    {
      sub_D68D20((__int64)&v40, 0, (__int64)a2);
      v6 = *(_DWORD *)sub_28EF5C0(a1 + 32, (__int64)&v40);
      sub_D68D70(&v40);
    }
    return v6;
  }
  v38[0] = 0;
  v33 = a1 + 32;
  v38[1] = 0;
  v39 = a2;
  v34 = a2 + 4096 != 0 && a2 + 0x2000 != 0;
  if ( v34 )
    sub_BD73F0((__int64)v38);
  if ( (unsigned __int8)sub_28EE370(v33, (__int64)v38, &v36) )
  {
    v4 = (unsigned int *)(v36 + 24);
    v5 = v39;
    goto LABEL_6;
  }
  v26 = *(_DWORD *)(a1 + 56);
  v27 = *(_DWORD *)(a1 + 48);
  v28 = v36;
  ++*(_QWORD *)(a1 + 32);
  v29 = v27 + 1;
  v37 = v28;
  if ( 4 * v29 >= 3 * v26 )
  {
    v26 *= 2;
    goto LABEL_52;
  }
  if ( v26 - *(_DWORD *)(a1 + 52) - v29 <= v26 >> 3 )
  {
LABEL_52:
    sub_28EF240(v33, v26);
    sub_28EE370(v33, (__int64)v38, &v37);
    v28 = v37;
    v29 = *(_DWORD *)(a1 + 48) + 1;
  }
  *(_DWORD *)(a1 + 48) = v29;
  v40 = 0;
  v41 = 0;
  v42 = -4096;
  if ( *(_QWORD *)(v28 + 16) != -4096 )
    --*(_DWORD *)(a1 + 52);
  sub_D68D70(&v40);
  v30 = v39;
  v5 = *(_BYTE **)(v28 + 16);
  if ( v39 != v5 )
  {
    if ( v5 != 0 && v5 + 4096 != 0 && v5 != (_BYTE *)-8192LL )
      sub_BD60C0((_QWORD *)v28);
    *(_QWORD *)(v28 + 16) = v30;
    if ( v30 != 0 && v30 + 4096 != 0 && v30 != (_BYTE *)-8192LL )
      sub_BD73F0(v28);
    v5 = v39;
  }
  *(_DWORD *)(v28 + 24) = 0;
  v4 = (unsigned int *)(v28 + 24);
LABEL_6:
  v6 = *v4;
  if ( v5 + 4096 != 0 && v5 != 0 && v5 != (_BYTE *)-8192LL )
    sub_BD60C0(v38);
  if ( !v6 )
  {
    v7 = *(__int64 **)(v3 + 40);
    v8 = *(_DWORD *)(a1 + 24);
    v38[0] = v7;
    if ( v8 )
    {
      v9 = *(_QWORD *)(a1 + 8);
      v10 = 1;
      v11 = 0;
      v12 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v13 = v9 + 16LL * v12;
      v14 = *(__int64 **)v13;
      if ( v7 == *(__int64 **)v13 )
      {
LABEL_12:
        v35 = *(_DWORD *)(v13 + 8);
        goto LABEL_13;
      }
      while ( v14 != (__int64 *)-4096LL )
      {
        if ( !v11 && v14 == (__int64 *)-8192LL )
          v11 = (__int64 *)v13;
        v12 = (v8 - 1) & (v10 + v12);
        v13 = v9 + 16LL * v12;
        v14 = *(__int64 **)v13;
        if ( v7 == *(__int64 **)v13 )
          goto LABEL_12;
        v10 = (unsigned int)(v10 + 1);
      }
      if ( !v11 )
        v11 = (__int64 *)v13;
      v31 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v32 = v31 + 1;
      v40 = v11;
      if ( 4 * (v31 + 1) < 3 * v8 )
      {
        if ( v8 - *(_DWORD *)(a1 + 20) - v32 > v8 >> 3 )
        {
LABEL_63:
          *(_DWORD *)(a1 + 16) = v32;
          if ( *v11 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v11 = (__int64)v7;
          *((_DWORD *)v11 + 2) = 0;
          v35 = 0;
LABEL_13:
          if ( (*(_DWORD *)(v3 + 4) & 0x7FFFFFF) != 0 )
          {
            v15 = 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF);
            v16 = 0;
            v17 = v3;
            v18 = 0;
            v19 = a1;
            v20 = v15;
            while ( v18 != v35 )
            {
              if ( (*(_BYTE *)(v17 + 7) & 0x40) != 0 )
                v21 = *(_QWORD *)(v17 - 8);
              else
                v21 = v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF);
              v22 = sub_28EF780(v19, *(_QWORD *)(v21 + v16), v7, v10);
              if ( v18 < v22 )
                v18 = v22;
              v16 += 32;
              if ( v16 == v20 )
              {
                v23 = v18;
                v3 = v17;
                v35 = v23;
                goto LABEL_22;
              }
            }
            v3 = v17;
          }
          else
          {
            v35 = 0;
          }
LABEL_22:
          v24 = *(_BYTE *)v3;
          v38[0] = 0;
          if ( v24 == 59 )
          {
            if ( (unsigned __int8)sub_28ECD10(v38, v3) )
              goto LABEL_47;
            v24 = *(_BYTE *)v3;
          }
          v40 = 0;
          if ( (v24 != 44 || !(unsigned __int8)sub_28EB290(&v40, v3))
            && !(unsigned __int8)sub_28EE9F0((unsigned __int8 *)v3) )
          {
            v6 = v35 + 1;
LABEL_26:
            v40 = 0;
            v41 = 0;
            v42 = v3;
            if ( v34 )
              sub_BD73F0((__int64)&v40);
            *(_DWORD *)sub_28EF5C0(v33, (__int64)&v40) = v6;
            sub_D68D70(&v40);
            return v6;
          }
LABEL_47:
          v6 = v35;
          goto LABEL_26;
        }
        sub_B23080(a1, v8);
LABEL_69:
        sub_B1C700(a1, (__int64 *)v38, &v40);
        v7 = v38[0];
        v11 = v40;
        v32 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_63;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
      v40 = 0;
    }
    sub_B23080(a1, 2 * v8);
    goto LABEL_69;
  }
  return v6;
}

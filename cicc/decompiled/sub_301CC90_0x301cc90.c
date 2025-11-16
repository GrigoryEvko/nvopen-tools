// Function: sub_301CC90
// Address: 0x301cc90
//
void __fastcall sub_301CC90(__int64 a1, __int64 a2, unsigned __int16 a3)
{
  unsigned __int64 v3; // r15
  int v4; // eax
  __int64 v5; // rax
  __int64 v6; // rbx
  char (__fastcall *v7)(__int64, __int64); // rax
  int v8; // eax
  __int64 *v9; // rbx
  __int64 *v10; // r12
  __int64 v11; // rdi
  int v12; // eax
  char v13; // al
  __int64 v14; // rbx
  unsigned __int8 *v15; // rsi
  __int64 v16; // rbx
  bool v17; // zf
  _QWORD *v18; // r12
  __int64 v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  const __m128i *v24; // r14
  const __m128i *v25; // r13
  const __m128i *v26; // rdx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rax
  _QWORD *v32; // [rsp+28h] [rbp-D8h]
  _QWORD *v33; // [rsp+30h] [rbp-D0h]
  unsigned __int8 *v35; // [rsp+48h] [rbp-B8h] BYREF
  unsigned __int8 *v36; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v37; // [rsp+58h] [rbp-A8h]
  __int64 v38; // [rsp+60h] [rbp-A0h]
  __m128i v39; // [rsp+70h] [rbp-90h] BYREF
  __int64 v40; // [rsp+80h] [rbp-80h]
  __int64 v41; // [rsp+88h] [rbp-78h]
  __int64 *v42; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v43; // [rsp+A8h] [rbp-58h]
  _BYTE v44[80]; // [rsp+B0h] [rbp-50h] BYREF

  v42 = (__int64 *)v44;
  v43 = 0x400000000LL;
  v32 = *(_QWORD **)(a1 + 328);
  if ( v32 == (_QWORD *)(a1 + 320) )
    return;
  do
  {
    v33 = v32 + 6;
    v3 = sub_2E313E0((__int64)v32);
    if ( v32 + 6 != (_QWORD *)v3 )
    {
      while ( 1 )
      {
        v4 = *(_DWORD *)(v3 + 44);
        if ( (v4 & 4) != 0 || (v4 & 8) == 0 )
          v5 = (*(_QWORD *)(*(_QWORD *)(v3 + 16) + 24LL) >> 5) & 1LL;
        else
          LOBYTE(v5) = sub_2E88A90(v3, 32, 1);
        v6 = 0;
        if ( (_BYTE)v5 )
        {
          v6 = 37;
          if ( !HIBYTE(a3) && *(unsigned __int16 *)(v3 + 68) != *(_DWORD *)(a2 + 76) )
            v6 = 0;
        }
        v7 = *(char (__fastcall **)(__int64, __int64))(*(_QWORD *)a2 + 1328LL);
        if ( v7 != sub_2FDE950 )
          break;
        v8 = *(_DWORD *)(v3 + 44);
        if ( (v8 & 4) != 0 || (v8 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v3 + 16) + 24LL) & 0x20LL) == 0 )
            goto LABEL_16;
        }
        else if ( !sub_2E88A90(v3, 32, 1) )
        {
          goto LABEL_16;
        }
        v12 = *(_DWORD *)(v3 + 44);
        if ( (v12 & 4) != 0 || (v12 & 8) == 0 )
          v13 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v3 + 16) + 24LL) >> 7;
        else
          v13 = sub_2E88A90(v3, 128, 1);
        if ( v13 )
          goto LABEL_35;
LABEL_16:
        if ( (_DWORD)v6 )
        {
          v14 = -40 * v6;
LABEL_37:
          v15 = *(unsigned __int8 **)(v3 + 56);
          v16 = *(_QWORD *)(a2 + 8) + v14;
          v35 = v15;
          if ( v15 )
          {
            sub_B96E90((__int64)&v35, (__int64)v15, 1);
            v36 = v35;
            if ( v35 )
            {
              sub_B976B0((__int64)&v35, v35, (__int64)&v36);
              v17 = (*(_BYTE *)(v3 + 44) & 4) == 0;
              v35 = 0;
              v37 = 0;
              v38 = 0;
              v18 = (_QWORD *)v32[4];
              v39.m128i_i64[0] = (__int64)v36;
              if ( !v17 )
              {
                if ( v36 )
                  sub_B96E90((__int64)&v39, (__int64)v36, 1);
LABEL_42:
                v19 = (__int64)sub_2E7B380(v18, v16, (unsigned __int8 **)&v39, 0);
                if ( v39.m128i_i64[0] )
                  sub_B91220((__int64)&v39, v39.m128i_i64[0]);
                sub_2E326B0((__int64)v32, (__int64 *)v3, v19);
                v20 = v37;
                if ( v37 )
LABEL_45:
                  sub_2E882B0(v19, (__int64)v18, v20);
LABEL_46:
                if ( v38 )
                  sub_2E88680(v19, (__int64)v18, v38);
                v21 = *(unsigned __int16 *)(v3 + 68);
                v39.m128i_i64[0] = 1;
                v40 = 0;
                v41 = v21;
                sub_2E8EAD0(v19, (__int64)v18, &v39);
                if ( v36 )
                  sub_B91220((__int64)&v36, (__int64)v36);
                if ( v35 )
                  sub_B91220((__int64)&v35, (__int64)v35);
                v24 = *(const __m128i **)(v3 + 32);
                v25 = (const __m128i *)((char *)v24 + 40 * (*(_DWORD *)(v3 + 40) & 0xFFFFFF));
                while ( v25 != v24 )
                {
                  v26 = v24;
                  v24 = (const __m128i *)((char *)v24 + 40);
                  sub_2E8EAD0(v19, (__int64)v18, v26);
                }
                v27 = (unsigned int)v43;
                v28 = (unsigned int)v43 + 1LL;
                if ( v28 > HIDWORD(v43) )
                {
                  sub_C8D5F0((__int64)&v42, v44, v28, 8u, v22, v23);
                  v27 = (unsigned int)v43;
                }
                v42[v27] = v3;
                LODWORD(v43) = v43 + 1;
                if ( sub_2E88F60(v3) )
                  sub_2E79700(a1, v3);
                goto LABEL_17;
              }
              if ( v36 )
                sub_B96E90((__int64)&v39, (__int64)v36, 1);
LABEL_67:
              v19 = (__int64)sub_2E7B380(v18, v16, (unsigned __int8 **)&v39, 0);
              if ( v39.m128i_i64[0] )
                sub_B91220((__int64)&v39, v39.m128i_i64[0]);
              sub_2E31040(v32 + 5, v19);
              v29 = *(_QWORD *)v3;
              v30 = *(_QWORD *)v19;
              *(_QWORD *)(v19 + 8) = v3;
              v29 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)v19 = v29 | v30 & 7;
              *(_QWORD *)(v29 + 8) = v19;
              *(_QWORD *)v3 = v19 | *(_QWORD *)v3 & 7LL;
              v20 = v37;
              if ( v37 )
                goto LABEL_45;
              goto LABEL_46;
            }
          }
          else
          {
            v36 = 0;
          }
          v37 = 0;
          v38 = 0;
          v18 = (_QWORD *)v32[4];
          if ( (*(_BYTE *)(v3 + 44) & 4) != 0 )
          {
            v39.m128i_i64[0] = 0;
            goto LABEL_42;
          }
          v39.m128i_i64[0] = 0;
          goto LABEL_67;
        }
LABEL_17:
        if ( (*(_BYTE *)v3 & 4) != 0 )
        {
          v3 = *(_QWORD *)(v3 + 8);
          if ( v33 == (_QWORD *)v3 )
            goto LABEL_19;
        }
        else
        {
          while ( (*(_BYTE *)(v3 + 44) & 8) != 0 )
            v3 = *(_QWORD *)(v3 + 8);
          v3 = *(_QWORD *)(v3 + 8);
          if ( v33 == (_QWORD *)v3 )
            goto LABEL_19;
        }
      }
      if ( !v7(a2, v3) )
        goto LABEL_16;
LABEL_35:
      if ( (_BYTE)a3 )
      {
        v14 = -1560;
        goto LABEL_37;
      }
      goto LABEL_16;
    }
LABEL_19:
    v32 = (_QWORD *)v32[1];
  }
  while ( (_QWORD *)(a1 + 320) != v32 );
  v9 = v42;
  v10 = &v42[(unsigned int)v43];
  if ( v42 != v10 )
  {
    do
    {
      v11 = *v9++;
      sub_2E88E20(v11);
    }
    while ( v10 != v9 );
    v10 = v42;
  }
  if ( v10 != (__int64 *)v44 )
    _libc_free((unsigned __int64)v10);
}

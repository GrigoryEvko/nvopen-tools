// Function: sub_23DE110
// Address: 0x23de110
//
void __fastcall sub_23DE110(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        __int64 *a6,
        __int64 a7)
{
  unsigned __int64 v7; // r12
  int v9; // edx
  __int64 v10; // rax
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rbx
  unsigned __int64 i; // rax
  char v16; // cl
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rbx
  __int16 v24; // ax
  _QWORD *v25; // rax
  __int64 v26; // r9
  __int64 v27; // r14
  __int64 v28; // rbx
  __int64 v29; // r12
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // r15
  __int64 v33; // rbx
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // rdx
  int v37; // r14d
  __int64 v38; // r14
  __int64 v39; // r12
  __int64 v40; // rdx
  unsigned int v41; // esi
  __int64 v42; // [rsp+0h] [rbp-E0h]
  unsigned __int64 v44; // [rsp+10h] [rbp-D0h]
  __int64 v45; // [rsp+28h] [rbp-B8h]
  __int64 v46; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v47; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v48; // [rsp+28h] [rbp-B8h]
  unsigned int v49; // [rsp+30h] [rbp-B0h]
  __int64 v50; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v51; // [rsp+30h] [rbp-B0h]
  char v53; // [rsp+45h] [rbp-9Bh]
  __int16 v54; // [rsp+46h] [rbp-9Ah]
  _BYTE v56[32]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v57; // [rsp+70h] [rbp-70h]
  _BYTE v58[32]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v59; // [rsp+A0h] [rbp-40h]

  if ( a4 < a5 )
  {
    v7 = a4;
    v9 = *(_DWORD *)(a1[1] + 80);
    v10 = 8;
    if ( (unsigned __int64)(v9 / 8) < 8 )
      v10 = v9 / 8;
    v44 = v10;
    v53 = *(_BYTE *)sub_B2BEC0(*a1);
    do
    {
      while ( !*(_BYTE *)(a2 + v7) )
      {
        if ( a5 <= ++v7 )
          return;
      }
      v11 = v44;
      if ( v44 <= a5 - v7 )
      {
        v11 = v44;
        v12 = v44 - 1;
        if ( v44 == 1 )
          goto LABEL_44;
      }
      else
      {
        do
          v11 >>= 1;
        while ( v11 > a5 - v7 );
        v12 = v11 - 1;
        if ( v11 == 1 )
        {
LABEL_44:
          v49 = 8;
          v11 = 1;
LABEL_14:
          v14 = 0;
          for ( i = 0; i < v11; ++i )
          {
            while ( 1 )
            {
              v17 = *(unsigned __int8 *)(v7 + a3 + i);
              if ( v53 )
                break;
              v16 = 8 * i++;
              v14 |= v17 << v16;
              if ( i >= v11 )
                goto LABEL_18;
            }
            v14 = v17 | (v14 << 8);
          }
          goto LABEL_18;
        }
      }
      do
      {
        if ( *(_BYTE *)(a2 + v7 + v12) )
          break;
        do
        {
          v13 = v11;
          v11 >>= 1;
        }
        while ( v11 >= v12 );
        v11 = v13;
        --v12;
      }
      while ( v12 );
      v49 = 8 * v11;
      if ( v11 )
        goto LABEL_14;
      v14 = 0;
LABEL_18:
      v57 = 257;
      v45 = sub_AD64C0(a1[58], v7, 0);
      v18 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)a6[10] + 32LL))(
              a6[10],
              13,
              a7,
              v45,
              0,
              0);
      if ( !v18 )
      {
        v59 = 257;
        v18 = sub_B504D0(13, a7, v45, (__int64)v58, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a6[11] + 16LL))(
          a6[11],
          v18,
          v56,
          a6[7],
          a6[8]);
        if ( *a6 != *a6 + 16LL * *((unsigned int *)a6 + 2) )
        {
          v47 = v11;
          v32 = *a6 + 16LL * *((unsigned int *)a6 + 2);
          v42 = v14;
          v33 = *a6;
          do
          {
            v34 = *(_QWORD *)(v33 + 8);
            v35 = *(_DWORD *)v33;
            v33 += 16;
            sub_B99FD0(v18, v35, v34);
          }
          while ( v32 != v33 );
          v11 = v47;
          v14 = v42;
        }
      }
      v19 = sub_BCD140((_QWORD *)a6[9], v49);
      v20 = sub_ACD640(v19, v14, 0);
      v57 = 257;
      v50 = v20;
      v21 = (__int64 *)sub_BD5C60(v20);
      v22 = sub_BCE3C0(v21, 0);
      if ( v22 == *(_QWORD *)(v18 + 8) )
      {
        v23 = v18;
      }
      else
      {
        v46 = v22;
        v23 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a6[10] + 120LL))(
                a6[10],
                48,
                v18,
                v22);
        if ( !v23 )
        {
          v59 = 257;
          v23 = sub_B51D30(48, v18, v46, (__int64)v58, 0, 0);
          if ( (unsigned __int8)sub_920620(v23) )
          {
            v36 = a6[12];
            v37 = *((_DWORD *)a6 + 26);
            if ( v36 )
              sub_B99FD0(v23, 3u, v36);
            sub_B45150(v23, v37);
          }
          (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a6[11] + 16LL))(
            a6[11],
            v23,
            v56,
            a6[7],
            a6[8]);
          if ( *a6 != *a6 + 16LL * *((unsigned int *)a6 + 2) )
          {
            v48 = v7;
            v38 = *a6 + 16LL * *((unsigned int *)a6 + 2);
            v39 = *a6;
            do
            {
              v40 = *(_QWORD *)(v39 + 8);
              v41 = *(_DWORD *)v39;
              v39 += 16;
              sub_B99FD0(v23, v41, v40);
            }
            while ( v38 != v39 );
            v7 = v48;
          }
        }
      }
      HIBYTE(v24) = HIBYTE(v54);
      v59 = 257;
      LOBYTE(v24) = 0;
      v54 = v24;
      v25 = sub_BD2C40(80, unk_3F10A10);
      v27 = (__int64)v25;
      if ( v25 )
        sub_B4D3C0((__int64)v25, v50, v23, 0, v54, v26, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a6[11] + 16LL))(
        a6[11],
        v27,
        v58,
        a6[7],
        a6[8]);
      if ( *a6 != *a6 + 16LL * *((unsigned int *)a6 + 2) )
      {
        v51 = v7;
        v28 = *a6 + 16LL * *((unsigned int *)a6 + 2);
        v29 = *a6;
        do
        {
          v30 = *(_QWORD *)(v29 + 8);
          v31 = *(_DWORD *)v29;
          v29 += 16;
          sub_B99FD0(v27, v31, v30);
        }
        while ( v28 != v29 );
        v7 = v51;
      }
      v7 += v11;
    }
    while ( a5 > v7 );
  }
}

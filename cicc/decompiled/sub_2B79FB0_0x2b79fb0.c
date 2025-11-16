// Function: sub_2B79FB0
// Address: 0x2b79fb0
//
void __fastcall sub_2B79FB0(__int64 **a1, _QWORD *a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int64 v9; // r14
  int v10; // r13d
  _DWORD *v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // r12
  __int64 *v16; // r14
  _BYTE *v17; // r12
  __int64 v18; // r14
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rsi
  unsigned int v22; // eax
  __int64 v23; // rdx
  unsigned int v24; // eax
  unsigned int v25; // eax
  unsigned int v26; // edx
  __int64 v27; // rcx
  _QWORD *v28; // rax
  __int64 v29; // r15
  __int64 v30; // r14
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // rax
  __int64 v34; // r12
  __int64 v35; // [rsp+0h] [rbp-120h]
  __int64 v36; // [rsp+8h] [rbp-118h]
  __int64 v37; // [rsp+10h] [rbp-110h]
  _QWORD *v38; // [rsp+18h] [rbp-108h]
  int v40; // [rsp+34h] [rbp-ECh]
  _BYTE *v41; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v42[4]; // [rsp+50h] [rbp-D0h] BYREF
  __int16 v43; // [rsp+70h] [rbp-B0h]
  _BYTE v44[32]; // [rsp+80h] [rbp-A0h] BYREF
  __int16 v45; // [rsp+A0h] [rbp-80h]
  void *s; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v47; // [rsp+B8h] [rbp-68h]
  _DWORD v48[24]; // [rsp+C0h] [rbp-60h] BYREF

  v7 = *(_QWORD *)(*a2 + 8LL);
  v8 = *(_QWORD *)(*a3 + 8LL);
  if ( v7 != v8 )
  {
    v9 = *(int *)(v7 + 32);
    v10 = *(_DWORD *)(v8 + 32);
    v40 = v9;
    if ( (int)v9 < v10 )
    {
      v9 = v10;
      v10 = v40;
    }
    s = v48;
    v47 = 0xC00000000LL;
    if ( v9 > 0xC )
    {
      sub_C8D5F0((__int64)&s, v48, v9, 4u, a5, a6);
      v11 = s;
      if ( 4 * v9 )
      {
        memset(s, 255, 4 * v9);
        v11 = s;
      }
      LODWORD(v47) = v9;
    }
    else
    {
      if ( v9 )
      {
        v22 = 4 * v9;
        if ( 4 * v9 )
        {
          if ( v22 < 8 )
          {
            if ( (v22 & 4) != 0 )
            {
              v48[0] = -1;
              v48[v22 / 4 - 1] = -1;
            }
            else if ( v22 )
            {
              LOBYTE(v48[0]) = -1;
            }
          }
          else
          {
            v23 = v22;
            v24 = v22 - 1;
            *(_QWORD *)((char *)&v48[-2] + v23) = -1;
            if ( v24 >= 8 )
            {
              v25 = v24 & 0xFFFFFFF8;
              v26 = 0;
              do
              {
                v27 = v26;
                v26 += 8;
                *(_QWORD *)((char *)v48 + v27) = -1;
              }
              while ( v26 < v25 );
            }
          }
        }
      }
      LODWORD(v47) = v9;
      v11 = v48;
    }
    if ( 4LL * v10 )
    {
      v12 = 0;
      do
      {
        v13 = v12;
        v11[v12] = v12;
        ++v12;
      }
      while ( (unsigned __int64)(4LL * v10 - 4) >> 2 != v13 );
      v11 = s;
    }
    v14 = a2;
    v15 = (unsigned int)v47;
    v43 = 257;
    if ( v40 != v10 )
      v14 = a3;
    v16 = *a1;
    v35 = (unsigned int)v47;
    v38 = v14;
    v37 = *v14;
    v36 = sub_ACADE0(*(__int64 ***)(*v14 + 8LL));
    v17 = (_BYTE *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, _DWORD *, __int64))(*(_QWORD *)v16[10] + 112LL))(
                     v16[10],
                     v37,
                     v36,
                     v11,
                     v15);
    if ( !v17 )
    {
      v45 = 257;
      v28 = sub_BD2C40(112, unk_3F1FE60);
      v17 = v28;
      if ( v28 )
        sub_B4E9E0((__int64)v28, v37, v36, v11, v35, (__int64)v44, 0, 0);
      (*(void (__fastcall **)(__int64, _BYTE *, __int64 *, __int64, __int64))(*(_QWORD *)v16[11] + 16LL))(
        v16[11],
        v17,
        v42,
        v16[7],
        v16[8]);
      v29 = *v16;
      v30 = *v16 + 16LL * *((unsigned int *)v16 + 2);
      while ( v30 != v29 )
      {
        v31 = *(_QWORD *)(v29 + 8);
        v32 = *(_DWORD *)v29;
        v29 += 16;
        sub_B99FD0((__int64)v17, v32, v31);
      }
    }
    *v38 = v17;
    if ( *v17 > 0x1Cu )
    {
      v41 = v17;
      v18 = (__int64)a1[1];
      sub_2400480((__int64)v44, v18, (__int64 *)&v41);
      if ( (_BYTE)v45 )
      {
        v33 = *(unsigned int *)(v18 + 40);
        v34 = (__int64)v41;
        if ( v33 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 44) )
        {
          sub_C8D5F0(v18 + 32, (const void *)(v18 + 48), v33 + 1, 8u, v19, v20);
          v33 = *(unsigned int *)(v18 + 40);
        }
        *(_QWORD *)(*(_QWORD *)(v18 + 32) + 8 * v33) = v34;
        ++*(_DWORD *)(v18 + 40);
      }
      v21 = (__int64)a1[2];
      v42[0] = *((_QWORD *)v41 + 5);
      sub_29B09C0((__int64)v44, v21, v42);
      v17 = (_BYTE *)*v38;
    }
    if ( v40 == v10 )
      *a2 = v17;
    else
      *a3 = v17;
    if ( s != v48 )
      _libc_free((unsigned __int64)s);
  }
}

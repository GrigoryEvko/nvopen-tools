// Function: sub_2511870
// Address: 0x2511870
//
__int64 __fastcall sub_2511870(
        __int64 a1,
        __int64 a2,
        unsigned __int8 *a3,
        __int64 **a4,
        __int64 *a5,
        _BYTE *a6,
        unsigned __int64 *a7)
{
  unsigned __int8 *v10; // r8
  int v12; // eax
  __int64 v13; // rcx
  int v14; // esi
  unsigned int v15; // eax
  __int64 v16; // rdx
  unsigned __int8 *v17; // rdi
  _BYTE **v18; // rdi
  __int64 v19; // r8
  __int64 v20; // r9
  _BYTE *v21; // rcx
  __int64 v22; // rsi
  __int64 i; // r9
  void (__fastcall *v24)(_BYTE *, __int64, __int64); // rax
  __int64 v25; // rax
  unsigned __int64 v26; // r12
  __int64 v27; // rdx
  _BYTE *v28; // r14
  void (__fastcall *v29)(_BYTE *, _BYTE *, __int64); // rax
  __int64 v30; // rax
  bool v31; // al
  char v32; // al
  unsigned __int8 v33; // al
  unsigned __int8 *v34; // r9
  unsigned int v35; // ebx
  int j; // edx
  int v37; // edx
  int v38; // r8d
  __int64 v39; // [rsp+8h] [rbp-A8h]
  int v40; // [rsp+14h] [rbp-9Ch]
  __int64 v41; // [rsp+18h] [rbp-98h]
  int v42; // [rsp+20h] [rbp-90h]
  _BYTE *v43; // [rsp+20h] [rbp-90h]
  unsigned __int8 *v44; // [rsp+20h] [rbp-90h]
  unsigned __int8 *v45; // [rsp+20h] [rbp-90h]
  __int64 v47; // [rsp+28h] [rbp-88h]
  char v48; // [rsp+37h] [rbp-79h] BYREF
  _QWORD v49[3]; // [rsp+38h] [rbp-78h] BYREF
  _BYTE *v50; // [rsp+50h] [rbp-60h] BYREF
  __int64 v51; // [rsp+58h] [rbp-58h]
  _BYTE v52[80]; // [rsp+60h] [rbp-50h] BYREF

  if ( *a3 != 60 )
  {
    v10 = (unsigned __int8 *)sub_D5D1D0(a3, a5, a4);
    if ( !v10 && *a3 == 3 )
    {
      v12 = *(_DWORD *)(a1 + 88);
      v13 = *(_QWORD *)(a1 + 72);
      v48 = 0;
      if ( v12 )
      {
        v14 = v12 - 1;
        v15 = (v12 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v16 = v13 + 56LL * v15;
        v17 = *(unsigned __int8 **)v16;
        if ( a3 == *(unsigned __int8 **)v16 )
        {
LABEL_7:
          v18 = &v50;
          v50 = v52;
          v51 = 0x100000000LL;
          v19 = *(unsigned int *)(v16 + 16);
          if ( (_DWORD)v19 && &v50 != (_BYTE **)(v16 + 8) )
          {
            v20 = 1;
            v21 = v52;
            if ( (_DWORD)v19 != 1 )
            {
              v41 = v16;
              v42 = *(_DWORD *)(v16 + 16);
              sub_2511770((__int64)&v50, (unsigned int)v19, v16, (__int64)v52, v19, 1);
              v16 = v41;
              v21 = v50;
              LODWORD(v19) = v42;
              v20 = *(unsigned int *)(v41 + 16);
            }
            v22 = *(_QWORD *)(v16 + 8);
            for ( i = v22 + 32 * v20; i != v22; v21 += 32 )
            {
              if ( v21 )
              {
                *((_QWORD *)v21 + 2) = 0;
                v24 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(v22 + 16);
                if ( v24 )
                {
                  v39 = i;
                  v18 = (_BYTE **)v21;
                  v40 = v19;
                  v43 = v21;
                  v24(v21, v22, 2);
                  v21 = v43;
                  i = v39;
                  LODWORD(v19) = v40;
                  *((_QWORD *)v43 + 3) = *(_QWORD *)(v22 + 24);
                  *((_QWORD *)v43 + 2) = *(_QWORD *)(v22 + 16);
                }
              }
              v22 += 32;
            }
            LODWORD(v51) = v19;
            v49[0] = a2;
            if ( !*((_QWORD *)v50 + 2) )
              sub_4263D6(v18, v22, v16);
            v25 = (*((__int64 (__fastcall **)(_BYTE *, unsigned __int8 *, _QWORD *, char *))v50 + 3))(
                    v50,
                    a3,
                    v49,
                    &v48);
            v26 = (unsigned __int64)v50;
            v49[1] = v25;
            v10 = (unsigned __int8 *)v25;
            v49[2] = v27;
            v28 = &v50[32 * (unsigned int)v51];
            if ( v50 != v28 )
            {
              do
              {
                v29 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)v28 - 2);
                v28 -= 32;
                if ( v29 )
                {
                  v44 = v10;
                  v29(v28, v28, 3);
                  v10 = v44;
                }
              }
              while ( (_BYTE *)v26 != v28 );
              v28 = v50;
            }
            if ( v28 != v52 )
            {
              v45 = v10;
              _libc_free((unsigned __int64)v28);
              v10 = v45;
            }
            if ( !v10 )
              return (__int64)v10;
            goto LABEL_28;
          }
        }
        else
        {
          v34 = *(unsigned __int8 **)v16;
          v35 = v14 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
          for ( j = 1; ; ++j )
          {
            if ( v34 == (unsigned __int8 *)-4096LL )
              goto LABEL_26;
            v35 = v14 & (v35 + j);
            v34 = *(unsigned __int8 **)(v13 + 56LL * v35);
            if ( a3 == v34 )
              break;
          }
          v37 = 1;
          while ( v17 != (unsigned __int8 *)-4096LL )
          {
            v38 = v37 + 1;
            v15 = v14 & (v37 + v15);
            v16 = v13 + 56LL * v15;
            v17 = *(unsigned __int8 **)v16;
            if ( v34 == *(unsigned __int8 **)v16 )
              goto LABEL_7;
            v37 = v38;
          }
        }
        BUG();
      }
LABEL_26:
      if ( (a3[32] & 0xFu) - 7 <= 1
        || (v31 = sub_B2FC80((__int64)a3), v10 = 0, !v31)
        && (v32 = sub_B2F6B0((__int64)a3), v10 = 0, !v32)
        && (v33 = a3[80], (v33 & 2) == 0)
        && (v33 & 1) != 0 )
      {
        v10 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
LABEL_28:
        if ( !a7 || *a7 == 0x7FFFFFFF || a7[1] == 0x7FFFFFFF )
        {
          return sub_96E500(v10, (__int64)a4, (__int64)a6);
        }
        else
        {
          v50 = (_BYTE *)*a7;
          LODWORD(v51) = 64;
          v30 = sub_9714E0((__int64)v10, (__int64)a4, (__int64)&v50, a6);
          v10 = (unsigned __int8 *)v30;
          if ( (unsigned int)v51 > 0x40 )
          {
            if ( v50 )
            {
              v47 = v30;
              j_j___libc_free_0_0((unsigned __int64)v50);
              return v47;
            }
          }
        }
      }
    }
    return (__int64)v10;
  }
  return sub_ACA8A0(a4);
}

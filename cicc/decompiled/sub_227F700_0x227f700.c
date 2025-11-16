// Function: sub_227F700
// Address: 0x227f700
//
__int64 __fastcall sub_227F700(__int64 a1, __int64 **a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 *v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rcx
  _QWORD *v22; // rbx
  __int64 v23; // r12
  _QWORD *v24; // rax
  _QWORD *v25; // rdi
  _QWORD *v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rcx
  _QWORD *v32; // r13
  _QWORD *v33; // rbx
  __int64 v34; // r12
  _QWORD *v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rsi
  void (__fastcall *v38)(_QWORD *, __int64, __int64, char *); // r8
  __int64 v39; // [rsp+8h] [rbp-E8h]
  __int64 *v41; // [rsp+18h] [rbp-D8h]
  _QWORD *v45; // [rsp+40h] [rbp-B0h]
  __int64 *i; // [rsp+48h] [rbp-A8h]
  __int64 v47; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v48; // [rsp+58h] [rbp-98h] BYREF
  char v49[8]; // [rsp+60h] [rbp-90h] BYREF
  unsigned __int64 v50; // [rsp+68h] [rbp-88h]
  char v51; // [rsp+7Ch] [rbp-74h]
  unsigned __int64 v52; // [rsp+98h] [rbp-58h]
  char v53; // [rsp+ACh] [rbp-44h]

  v7 = (__int64)a3;
  v8 = *(_QWORD *)(sub_227ED20(a4, &qword_4F8A320, a3, a5) + 8);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 2;
  v47 = v8;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  sub_AE6EC0(a1, (__int64)&unk_4F82400);
  v9 = sub_227B160(a4, (__int64)&qword_4FDADA8, v7);
  if ( !v9 )
    BUG();
  v39 = *(_QWORD *)(v9 + 8);
  v41 = a2[1];
  if ( v41 != *a2 )
  {
    for ( i = *a2; v41 != i; ++i )
    {
      if ( (unsigned __int8)sub_227B670(&v47, *i, v7) )
      {
        (*(void (__fastcall **)(char *, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*i + 16LL))(
          v49,
          *i,
          v7,
          a4,
          a5,
          a6);
        v18 = *(__int64 **)(a6 + 16);
        if ( v18 )
        {
          v7 = *(_QWORD *)(a6 + 16);
          *(_QWORD *)(sub_227ED20(a4, &qword_4FDADA8, v18, a5) + 8) = v39;
        }
        sub_227AD80(a1, (__int64)v49, v14, v15, v16, v17);
        if ( (unsigned __int8)sub_B19060(*(_QWORD *)(a6 + 8), v7, v19, v20) )
        {
          if ( v47 )
          {
            v32 = *(_QWORD **)(v47 + 576);
            v33 = &v32[4 * *(unsigned int *)(v47 + 584)];
            if ( v32 != v33 )
            {
              v34 = *i;
              do
              {
                v35 = v32;
                v37 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v34 + 32LL))(v34);
                v38 = *(void (__fastcall **)(_QWORD *, __int64, __int64, char *))(v32[3] & 0xFFFFFFFFFFFFFFF8LL);
                if ( (v32[3] & 2) == 0 )
                  v35 = (_QWORD *)*v32;
                v32 += 4;
                v38(v35, v37, v36, v49);
              }
              while ( v33 != v32 );
            }
          }
          sub_227AD40((__int64)v49);
          break;
        }
        sub_227C930(a4, v7, (__int64)v49, v21);
        if ( v47 )
        {
          v22 = *(_QWORD **)(v47 + 432);
          v45 = &v22[4 * *(unsigned int *)(v47 + 440)];
          if ( v22 != v45 )
          {
            v23 = *i;
            do
            {
              v48 = 0;
              v24 = (_QWORD *)sub_22077B0(0x10u);
              if ( v24 )
              {
                v24[1] = v7;
                *v24 = &unk_4A08BA8;
              }
              v25 = v48;
              v48 = v24;
              if ( v25 )
                (*(void (__fastcall **)(_QWORD *))(*v25 + 8LL))(v25);
              v26 = v22;
              v28 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v23 + 32LL))(v23);
              if ( (v22[3] & 2) == 0 )
                v26 = (_QWORD *)*v22;
              (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD **, char *))(v22[3] & 0xFFFFFFFFFFFFFFF8LL))(
                v26,
                v28,
                v27,
                &v48,
                v49);
              if ( v48 )
                (*(void (__fastcall **)(_QWORD *))(*v48 + 8LL))(v48);
              v22 += 4;
            }
            while ( v45 != v22 );
          }
        }
        if ( !v53 )
          _libc_free(v52);
        if ( !v51 )
          _libc_free(v50);
      }
    }
  }
  sub_227AD80(a6 + 24, a1, v10, v11, v12, v13);
  if ( *(_DWORD *)(a1 + 68) != *(_DWORD *)(a1 + 72) || !(unsigned __int8)sub_B19060(a1, (__int64)&unk_4F82400, v29, v30) )
    sub_AE6EC0(a1, (__int64)&unk_4FDADC8);
  return a1;
}

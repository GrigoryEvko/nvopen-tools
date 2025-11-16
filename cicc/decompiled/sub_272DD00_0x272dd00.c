// Function: sub_272DD00
// Address: 0x272dd00
//
void __fastcall sub_272DD00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v8; // r13
  __int64 v9; // rsi
  __int64 v10; // r14
  unsigned int v11; // eax
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rdx
  unsigned __int64 v16; // rbx
  __int64 v17; // r15
  char *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rbx
  unsigned int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // r15
  char *v24; // rdi
  char *v26; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v27; // [rsp+28h] [rbp-D8h]
  _BYTE v28[128]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v29; // [rsp+B0h] [rbp-50h]
  __int64 v30; // [rsp+B8h] [rbp-48h]
  int v31; // [rsp+C0h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = a1 + 168;
    if ( a1 + 168 != a2 )
    {
      v8 = a1 + 168;
      do
      {
        v9 = *(_QWORD *)(a1 + 144);
        v10 = *(_QWORD *)(v8 + 144);
        if ( *(_QWORD *)(v10 + 8) == *(_QWORD *)(v9 + 8) )
          v11 = (unsigned int)sub_C49970(v10 + 24, (unsigned __int64 *)(v9 + 24)) >> 31;
        else
          LOBYTE(v11) = *(_DWORD *)(v10 + 32) < *(_DWORD *)(v9 + 32);
        v12 = *(unsigned int *)(v8 + 8);
        if ( (_BYTE)v11 )
        {
          v26 = v28;
          v27 = 0x800000000LL;
          if ( (_DWORD)v12 )
          {
            sub_272D8A0((__int64)&v26, (char **)v8, v6, v12, a5, a6);
            v10 = *(_QWORD *)(v8 + 144);
          }
          v13 = *(_QWORD *)(v8 + 152);
          v29 = v10;
          v14 = v8 + 168;
          v15 = 0xCF3CF3CF3CF3CF3DLL;
          v30 = v13;
          v31 = *(_DWORD *)(v8 + 160);
          v16 = 0xCF3CF3CF3CF3CF3DLL * ((v8 - a1) >> 3);
          if ( v8 - a1 > 0 )
          {
            do
            {
              v17 = v8;
              v8 -= 168;
              sub_272D8A0(v17, (char **)v8, v15, v12, a5, a6);
              *(_QWORD *)(v17 + 144) = *(_QWORD *)(v17 - 24);
              *(_QWORD *)(v17 + 152) = *(_QWORD *)(v17 - 16);
              v12 = *(unsigned int *)(v17 - 8);
              *(_DWORD *)(v17 + 160) = v12;
              --v16;
            }
            while ( v16 );
          }
          sub_272D8A0(a1, &v26, v15, v12, a5, a6);
          v18 = v26;
          *(_QWORD *)(a1 + 144) = v29;
          *(_QWORD *)(a1 + 152) = v30;
          *(_DWORD *)(a1 + 160) = v31;
          if ( v18 != v28 )
            _libc_free((unsigned __int64)v18);
        }
        else
        {
          v26 = v28;
          v27 = 0x800000000LL;
          if ( (_DWORD)v12 )
          {
            sub_272D8A0((__int64)&v26, (char **)v8, v6, v12, a5, a6);
            v10 = *(_QWORD *)(v8 + 144);
          }
          v19 = *(_QWORD *)(v8 + 152);
          v29 = v10;
          v20 = v8;
          v30 = v19;
          v31 = *(_DWORD *)(v8 + 160);
          while ( 1 )
          {
            v22 = *(_QWORD *)(v20 - 24);
            v23 = v20;
            if ( *(_QWORD *)(v10 + 8) == *(_QWORD *)(v22 + 8) )
              v21 = (unsigned int)sub_C49970(v10 + 24, (unsigned __int64 *)(v22 + 24)) >> 31;
            else
              LOBYTE(v21) = *(_DWORD *)(v10 + 32) < *(_DWORD *)(v22 + 32);
            v20 -= 168;
            if ( !(_BYTE)v21 )
              break;
            sub_272D8A0(v23, (char **)v20, v6, v12, a5, a6);
            v10 = v29;
            *(_QWORD *)(v20 + 312) = *(_QWORD *)(v20 + 144);
            *(_QWORD *)(v20 + 320) = *(_QWORD *)(v20 + 152);
            *(_DWORD *)(v20 + 328) = *(_DWORD *)(v20 + 160);
          }
          sub_272D8A0(v23, &v26, v6, v12, a5, a6);
          v24 = v26;
          *(_QWORD *)(v23 + 144) = v29;
          *(_QWORD *)(v23 + 152) = v30;
          *(_DWORD *)(v23 + 160) = v31;
          if ( v24 != v28 )
            _libc_free((unsigned __int64)v24);
          v14 = v8 + 168;
        }
        v8 = v14;
      }
      while ( v14 != a2 );
    }
  }
}

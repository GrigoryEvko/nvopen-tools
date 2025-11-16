// Function: sub_264AF10
// Address: 0x264af10
//
void __fastcall sub_264AF10(__int64 a1, __int64 a2, char a3, __m128i a4)
{
  int v4; // eax
  unsigned __int64 v5; // rbx
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // r14
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  char v17; // dl
  unsigned __int8 v18; // al
  __int64 v19; // rdx
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // r12
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  unsigned __int8 v25; // al
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v29; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v30; // [rsp+20h] [rbp-B0h] BYREF
  unsigned __int8 v31; // [rsp+28h] [rbp-A8h]
  __m128i v32[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD *v33[2]; // [rsp+50h] [rbp-80h] BYREF
  char v34; // [rsp+60h] [rbp-70h]
  const __m128i *v35[2]; // [rsp+70h] [rbp-60h] BYREF
  __int64 v36; // [rsp+90h] [rbp-40h]

  v4 = dword_4FF3E28;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  if ( v4 == 1 )
  {
    if ( !(unsigned int)sub_23DF0D0(&dword_4FF3CC8) )
      sub_C64ED0("-memprof-dot-scope=alloc requires -memprof-dot-alloc-id", 1u);
    v4 = dword_4FF3E28;
  }
  if ( v4 == 2 )
  {
    if ( !(unsigned int)sub_23DF0D0(&dword_4FF3BE8) )
      sub_C64ED0("-memprof-dot-scope=context requires -memprof-dot-context-id", 1u);
    v4 = dword_4FF3E28;
  }
  if ( !v4 && (unsigned int)sub_23DF0D0(&dword_4FF3CC8) && (unsigned int)sub_23DF0D0(&dword_4FF3BE8) )
    sub_C64ED0("-memprof-dot-scope=all can't have both -memprof-dot-alloc-id and -memprof-dot-context-id", 1u);
  if ( !*(_QWORD *)a1 && qword_4FF38D0 )
  {
    LOWORD(v36) = 260;
    v35[0] = (const __m128i *)&qword_4FF38C8;
    sub_C7EA90((__int64)v33, (__int64 *)v35, 0, 1u, 0, 0);
    if ( (v34 & 1) != 0 && LODWORD(v33[0]) )
    {
      sub_C63CA0(v32[0].m128i_i64, (int)v33[0], (__int64)v33[1]);
      v5 = v32[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v34 & 1) == 0 )
      {
        if ( v33[0] )
          (*(void (__fastcall **)(_QWORD *))(*v33[0] + 8LL))(v33[0]);
      }
      sub_8FD6D0((__int64)v33, "Error loading file '", &qword_4FF38C8);
      sub_94F930(v32, (__int64)v33, "': ");
      v35[0] = v32;
      LOWORD(v36) = 260;
      v6 = (__int64 *)sub_CB72A0();
      v30 = v5 | 1;
      sub_C63F70((unsigned __int64 *)&v30, v6, v7, v8, v9, v10, (char)v35[0]);
      sub_9C66B0(&v30);
      sub_2240A30((unsigned __int64 *)v32);
      sub_2240A30((unsigned __int64 *)v33);
      return;
    }
    v11 = v33[0];
    v12 = (__int64)v33[0];
    sub_C7EC60(v35, v33[0]);
    sub_9F1E00((__int64)&v30, v12, v13, v14, v15, v16, a4, v35[0], (unsigned __int64)v35[1]);
    v17 = v31 & 1;
    v18 = (2 * (v31 & 1)) | v31 & 0xFD;
    v31 = v18;
    if ( v17 )
    {
      sub_8FD6D0((__int64)v33, "Error parsing file '", &qword_4FF38C8);
      sub_94F930(v32, (__int64)v33, "': ");
      v35[0] = v32;
      LOWORD(v36) = 260;
      v12 = (__int64)sub_CB72A0();
      v25 = v31;
      v26 = v31 & 0xFD;
      v31 &= ~2u;
      if ( (v25 & 1) != 0 )
      {
        v27 = v30;
        v30 = 0;
        v28 = v27 | 1;
      }
      else
      {
        v28 = 1;
        v29 = 0;
        sub_9C66B0(&v29);
      }
      sub_C63F70((unsigned __int64 *)&v28, (__int64 *)v12, v26, v22, v23, v24, (char)v35[0]);
      sub_9C66B0(&v28);
      sub_2240A30((unsigned __int64 *)v32);
      sub_2240A30((unsigned __int64 *)v33);
      v18 = v31;
      if ( (v31 & 2) == 0 )
        goto LABEL_23;
    }
    else
    {
      v19 = v30;
      v20 = *(_QWORD *)(a1 + 8);
      v30 = 0;
      *(_QWORD *)(a1 + 8) = v19;
      if ( !v20 )
      {
        *(_QWORD *)a1 = v19;
        goto LABEL_23;
      }
      sub_9CD560(v20);
      v12 = 584;
      j_j___libc_free_0(v20);
      v18 = v31;
      *(_QWORD *)a1 = *(_QWORD *)(a1 + 8);
      if ( (v18 & 2) == 0 )
      {
LABEL_23:
        v21 = v30;
        if ( (v18 & 1) != 0 )
        {
          if ( v30 )
            (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v30 + 8LL))(v30, v12);
        }
        else if ( v30 )
        {
          sub_9CD560(v30);
          v12 = 584;
          j_j___libc_free_0(v21);
        }
        if ( v11 )
          (*(void (__fastcall **)(_QWORD *, __int64))(*v11 + 8LL))(v11, v12);
        return;
      }
    }
    sub_25CE240(&v30, v12);
  }
}

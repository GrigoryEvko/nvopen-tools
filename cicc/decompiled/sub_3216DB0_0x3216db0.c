// Function: sub_3216DB0
// Address: 0x3216db0
//
__int64 __fastcall sub_3216DB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  void (__fastcall *v7)(__int64, unsigned __int64, _QWORD); // r14
  __int64 v8; // rdi
  unsigned __int64 v9; // rsi
  __int64 result; // rax
  __int64 v11; // r15
  _QWORD *v12; // r14
  __int64 v13; // rcx
  __int64 v14; // rax
  size_t v15; // rdx
  void (*v16)(); // rax
  __int64 *v17; // rax
  __int64 *v18; // rbx
  __int64 v19; // r14
  __int64 *v20; // r12
  __int64 v21; // rsi
  void (*v22)(); // rax
  void (*v23)(); // rax
  int v24; // ebx
  void (*v25)(); // rax
  void (*v26)(); // rax
  __int64 v27; // rbx
  __int64 v28; // r14
  unsigned int v29; // eax
  void (*v30)(); // rax
  _QWORD *v31; // [rsp+10h] [rbp-C0h]
  __int64 v33; // [rsp+20h] [rbp-B0h]
  unsigned int v34; // [rsp+30h] [rbp-A0h]
  unsigned int v35; // [rsp+34h] [rbp-9Ch]
  __int64 v36; // [rsp+38h] [rbp-98h]
  const char *v37; // [rsp+40h] [rbp-90h] BYREF
  char v38; // [rsp+60h] [rbp-70h]
  char v39; // [rsp+61h] [rbp-6Fh]
  _QWORD v40[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v41; // [rsp+90h] [rbp-40h]

  v6 = *(_QWORD *)(a4 + 224);
  v35 = sub_AE4380(a2 + 312, 0);
  v7 = *(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)v6 + 176LL);
  v8 = *(_QWORD *)(sub_31DA6B0(a4) + 920);
  v37 = ".note.gc";
  v39 = 1;
  v38 = 3;
  v41 = 257;
  v9 = sub_E71CB0(v8, (size_t *)&v37, 1, 0, 0, (__int64)v40, 0, -1, 0);
  v7(v6, v9, 0);
  result = *(_QWORD *)(a3 + 224);
  v33 = *(_QWORD *)(a3 + 232);
  if ( result != v33 )
  {
    v36 = *(_QWORD *)(a3 + 224);
    v11 = a4;
    v34 = (v35 != 4) + 5;
    do
    {
      v12 = *(_QWORD **)v36;
      v13 = *(_QWORD *)(a1 + 8);
      v14 = *(_QWORD *)(*(_QWORD *)v36 + 8LL);
      v15 = *(_QWORD *)(v14 + 16);
      if ( v15 == *(_QWORD *)(v13 + 16) )
      {
        if ( !v15 || (v9 = *(_QWORD *)(v13 + 8), !memcmp(*(const void **)(v14 + 8), (const void *)v9, v15)) )
        {
          LOBYTE(v9) = v35 != 4;
          sub_31DCA70(v11, v9 + 2, 0, 0);
          v16 = *(void (**)())(*(_QWORD *)v6 + 120LL);
          v40[0] = "safe point count";
          v41 = 259;
          if ( v16 != nullsub_98 )
            ((void (__fastcall *)(__int64, _QWORD *, __int64))v16)(v6, v40, 1);
          sub_31DC9F0(v11, (__int64)(v12[7] - v12[6]) >> 4);
          v17 = (__int64 *)v12[7];
          v18 = (__int64 *)v12[6];
          if ( v17 != v18 )
          {
            v31 = v12;
            v19 = v6;
            v20 = v17;
            do
            {
              v22 = *(void (**)())(*(_QWORD *)v19 + 120LL);
              v40[0] = "safe point address";
              v41 = 259;
              if ( v22 != nullsub_98 )
                ((void (__fastcall *)(__int64, _QWORD *, __int64))v22)(v19, v40, 1);
              v21 = *v18;
              v18 += 2;
              (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v11 + 432LL))(
                v11,
                v21,
                0,
                4,
                0);
            }
            while ( v20 != v18 );
            v6 = v19;
            v12 = v31;
          }
          v23 = *(void (**)())(*(_QWORD *)v6 + 120LL);
          v40[0] = "stack frame size (in words)";
          v41 = 259;
          if ( v23 != nullsub_98 )
            ((void (__fastcall *)(__int64, _QWORD *, __int64))v23)(v6, v40, 1);
          sub_31DC9F0(v11, v12[2] / (unsigned __int64)v35);
          v24 = *(_QWORD *)(*v12 + 104LL) - v34;
          if ( (unsigned __int64)v34 >= *(_QWORD *)(*v12 + 104LL) )
            v24 = 0;
          v25 = *(void (**)())(*(_QWORD *)v6 + 120LL);
          v40[0] = "stack arity";
          v41 = 259;
          if ( v25 != nullsub_98 )
            ((void (__fastcall *)(__int64, _QWORD *, __int64))v25)(v6, v40, 1);
          sub_31DC9F0(v11, v24);
          v26 = *(void (**)())(*(_QWORD *)v6 + 120LL);
          v40[0] = "live root count";
          v41 = 259;
          if ( v26 != nullsub_98 )
            ((void (__fastcall *)(__int64, _QWORD *, __int64))v26)(v6, v40, 1);
          v9 = (__int64)(v12[4] - v12[3]) >> 4;
          sub_31DC9F0(v11, v9);
          v27 = v12[3];
          v28 = v12[4];
          while ( v28 != v27 )
          {
            v30 = *(void (**)())(*(_QWORD *)v6 + 120LL);
            v40[0] = "stack index (offset / wordsize)";
            v41 = 259;
            if ( v30 != nullsub_98 )
              ((void (__fastcall *)(__int64, _QWORD *, __int64))v30)(v6, v40, 1);
            v29 = *(_DWORD *)(v27 + 4);
            v27 += 16;
            LODWORD(v9) = v29 / v35;
            sub_31DC9F0(v11, v29 / v35);
          }
        }
      }
      v36 += 8;
      result = v36;
    }
    while ( v33 != v36 );
  }
  return result;
}

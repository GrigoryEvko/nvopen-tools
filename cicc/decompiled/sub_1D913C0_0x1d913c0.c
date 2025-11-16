// Function: sub_1D913C0
// Address: 0x1d913c0
//
__int64 __fastcall sub_1D913C0(_QWORD *a1, __int64 a2)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 (*v14)(void); // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // r8
  __int64 v18; // rbx
  __int64 (*v19)(); // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r15
  __int64 (*v23)(void); // rax
  __int64 v24; // rcx
  char *v25; // r13
  char *i; // rsi
  char *v27; // rbx
  __int64 v28; // r13
  __int64 v29; // rbx
  __int64 v30; // r15
  __int16 v31; // ax
  __int16 v32; // ax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 *v35; // r10
  int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 *v39; // [rsp-68h] [rbp-68h]
  __int64 v40; // [rsp-58h] [rbp-58h]
  int v41; // [rsp-44h] [rbp-44h] BYREF
  __int64 v42[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(*(_QWORD *)a2 + 19LL) & 0x40) == 0 )
    return 0;
  v4 = (__int64 *)a1[1];
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_60:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4FC3606 )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_60;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4FC3606);
  v9 = sub_1D8F610(v8, *(_QWORD *)a2);
  v10 = (__int64 *)a1[1];
  a1[29] = v9;
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_61:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4FC6A0E )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_61;
  }
  a1[30] = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(
             *(_QWORD *)(v11 + 8),
             &unk_4FC6A0E);
  v14 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v15 = 0;
  if ( v14 != sub_1D00B00 )
    v15 = v14();
  a1[31] = v15;
  v16 = *(_QWORD *)(a2 + 16);
  v17 = 0;
  v18 = *(_QWORD *)(a2 + 56);
  v19 = *(__int64 (**)())(*(_QWORD *)v16 + 112LL);
  if ( v19 != sub_1D00B10 )
    v17 = ((__int64 (__fastcall *)(__int64, void *, __int64 (*)(void), __int64, _QWORD))v19)(
            v16,
            &unk_4FC6A0E,
            v14,
            v13,
            0);
  if ( *(_BYTE *)(v18 + 36) || (unsigned __int8)sub_1F4B450(v17, a2) )
  {
    v20 = a1[29];
    v21 = -1;
  }
  else
  {
    v20 = a1[29];
    v21 = *(_QWORD *)(v18 + 48);
  }
  *(_QWORD *)(v20 + 16) = v21;
  if ( *(_DWORD *)(*(_QWORD *)(a1[29] + 8LL) + 44LL) )
  {
    v28 = *(_QWORD *)(a2 + 328);
    if ( v28 != a2 + 320 )
    {
      while ( 1 )
      {
        v29 = *(_QWORD *)(v28 + 32);
        v30 = v28 + 24;
        if ( v29 != v28 + 24 )
          break;
LABEL_54:
        v28 = *(_QWORD *)(v28 + 8);
        if ( a2 + 320 == v28 )
          goto LABEL_18;
      }
      while ( 1 )
      {
        v31 = *(_WORD *)(v29 + 46);
        if ( (v31 & 4) != 0 || (v31 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v29 + 16) + 8LL) & 0x10LL) != 0 )
          {
LABEL_40:
            v32 = *(_WORD *)(v29 + 46);
            if ( (v32 & 4) != 0 || (v32 & 8) == 0 )
              v33 = (*(_QWORD *)(*(_QWORD *)(v29 + 16) + 8LL) >> 6) & 1LL;
            else
              LOBYTE(v33) = sub_1E15D00(v29, 64, 1);
            if ( !(_BYTE)v33 )
            {
              v34 = v29;
              if ( (*(_BYTE *)v29 & 4) == 0 && (*(_BYTE *)(v29 + 46) & 8) != 0 )
              {
                do
                  v34 = *(_QWORD *)(v34 + 8);
                while ( (*(_BYTE *)(v34 + 46) & 8) != 0 );
              }
              v35 = *(__int64 **)(v34 + 8);
              v36 = *(_DWORD *)(*(_QWORD *)(a1[29] + 8LL) + 44LL);
              if ( (v36 & 1) != 0 )
              {
                v39 = v35;
                v42[0] = sub_1D90200((__int64)a1, *(_QWORD *)(v29 + 24), (__int64 *)v29, v29 + 64);
                v38 = a1[29];
                v41 = 0;
                sub_1D91300((__int64 *)(v38 + 48), &v41, v42, (__int64 *)(v29 + 64));
                v35 = v39;
                v36 = *(_DWORD *)(*(_QWORD *)(a1[29] + 8LL) + 44LL);
              }
              if ( (v36 & 2) != 0 )
              {
                v42[0] = sub_1D90200((__int64)a1, *(_QWORD *)(v29 + 24), v35, v29 + 64);
                v37 = a1[29];
                v41 = 1;
                sub_1D91300((__int64 *)(v37 + 48), &v41, v42, (__int64 *)(v29 + 64));
              }
            }
          }
        }
        else if ( (unsigned __int8)sub_1E15D00(v29, 16, 1) )
        {
          goto LABEL_40;
        }
        if ( (*(_BYTE *)v29 & 4) != 0 )
        {
          v29 = *(_QWORD *)(v29 + 8);
          if ( v29 == v30 )
            goto LABEL_54;
        }
        else
        {
          while ( (*(_BYTE *)(v29 + 46) & 8) != 0 )
            v29 = *(_QWORD *)(v29 + 8);
          v29 = *(_QWORD *)(v29 + 8);
          if ( v29 == v30 )
            goto LABEL_54;
        }
      }
    }
  }
LABEL_18:
  v22 = 0;
  v23 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 48LL);
  if ( v23 != sub_1D90020 )
    v22 = v23();
  v24 = a1[29];
  v25 = *(char **)(v24 + 24);
  for ( i = *(char **)(v24 + 32); i != v25; i = *(char **)(v24 + 32) )
  {
    v27 = v25 + 16;
    if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 56) + 8LL)
                   + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a2 + 56) + 32LL) + *(_DWORD *)v25)
                   + 8) == -1 )
    {
      if ( v27 != i )
      {
        v40 = v24;
        memmove(v25, v25 + 16, i - v27);
        v24 = v40;
      }
      *(_QWORD *)(v24 + 32) -= 16LL;
    }
    else
    {
      *((_DWORD *)v25 + 1) = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64 *))(*(_QWORD *)v22 + 176LL))(
                               v22,
                               a2,
                               *(unsigned int *)v25,
                               v42);
      v25 += 16;
    }
    v24 = a1[29];
  }
  return 0;
}

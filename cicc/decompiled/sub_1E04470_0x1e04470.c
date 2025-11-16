// Function: sub_1E04470
// Address: 0x1e04470
//
__int64 __fastcall sub_1E04470(__int64 a1, __int64 a2)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  __int64 (*v5)(void); // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 (*v16)(void); // rdx
  int v17; // eax
  bool v18; // zf
  __int64 v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // r8d
  __int64 **v23; // r9
  __int64 *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdi

  if ( !(unsigned __int8)sub_1636880(a1, *(_QWORD *)a2) )
  {
    v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
    v4 = 0;
    if ( v3 != sub_1D00B00 )
      v4 = v3();
    *(_QWORD *)(a1 + 232) = v4;
    v5 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 112LL);
    v6 = 0;
    if ( v5 != sub_1D00B10 )
      v6 = v5();
    *(_QWORD *)(a1 + 240) = v6;
    v7 = *(__int64 **)(a1 + 8);
    *(_QWORD *)(a1 + 264) = *(_QWORD *)(a2 + 40);
    v8 = *v7;
    v9 = v7[1];
    if ( v8 != v9 )
    {
      while ( *(_UNKNOWN **)v8 != &unk_4F96DB4 )
      {
        v8 += 16;
        if ( v9 == v8 )
          goto LABEL_28;
      }
      v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(
              *(_QWORD *)(v8 + 8),
              &unk_4F96DB4);
      v11 = *(__int64 **)(a1 + 8);
      *(_QWORD *)(a1 + 248) = *(_QWORD *)(v10 + 160);
      v12 = *v11;
      v13 = v11[1];
      if ( v12 != v13 )
      {
        while ( *(_UNKNOWN **)v12 != &unk_4FC62EC )
        {
          v12 += 16;
          if ( v13 == v12 )
            goto LABEL_28;
        }
        v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(
                *(_QWORD *)(v12 + 8),
                &unk_4FC62EC);
        v15 = *(_QWORD *)(a1 + 232);
        *(_QWORD *)(a1 + 256) = v14;
        v16 = *(__int64 (**)(void))(*(_QWORD *)v15 + 960LL);
        v17 = 5;
        if ( v16 != sub_1DF72D0 )
          v17 = v16();
        v18 = byte_4FC62C0 == 0;
        *(_DWORD *)(a1 + 296) = v17;
        if ( v18 )
          goto LABEL_17;
        v25 = *(__int64 **)(a1 + 8);
        v26 = *v25;
        v27 = v25[1];
        if ( v26 != v27 )
        {
          while ( *(_UNKNOWN **)v26 != &unk_4FD4138 )
          {
            v26 += 16;
            if ( v27 == v26 )
              goto LABEL_28;
          }
          v28 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v26 + 8) + 104LL))(
                  *(_QWORD *)(v26 + 8),
                  &unk_4FD4138);
          v29 = *(__int64 **)(a1 + 8);
          *(_QWORD *)(a1 + 272) = *(_QWORD *)(v28 + 232);
          v30 = *v29;
          v31 = v29[1];
          if ( v30 != v31 )
          {
            while ( *(_UNKNOWN **)v30 != &unk_4FB9E2C )
            {
              v30 += 16;
              if ( v31 == v30 )
                goto LABEL_28;
            }
            v32 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v30 + 8) + 104LL))(
                    *(_QWORD *)(v30 + 8),
                    &unk_4FB9E2C);
            v33 = *(_QWORD *)(a1 + 272);
            *(_QWORD *)(a1 + 280) = v32 + 156;
            sub_1E03840(v33, *(_DWORD *)(v32 + 164), 0, 1);
LABEL_17:
            v19 = *(_QWORD *)(a1 + 256);
            sub_1E06620(v19);
            return sub_1E00370(a1, *(_QWORD *)(*(_QWORD *)(v19 + 1312) + 56LL), v20, v21, v22, v23);
          }
        }
      }
    }
LABEL_28:
    BUG();
  }
  return 0;
}

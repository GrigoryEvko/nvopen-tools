// Function: sub_39703E0
// Address: 0x39703e0
//
__int64 __fastcall sub_39703E0(_QWORD *a1, _DWORD *a2, __int64 a3, unsigned int a4)
{
  __int64 (*v7)(); // rax
  __int64 v8; // rax
  unsigned int *v9; // r13
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rdi
  __int64 (__fastcall *v17)(__int64, __int64); // r13
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 (*v24)(); // rax
  __int64 v25; // rax
  __int64 v26; // rax

  switch ( *a2 )
  {
    case 0:
      v10 = a1[31];
      v11 = sub_1DD5A70(a3);
      v9 = (unsigned int *)sub_38CF310(v11, 0, v10, 0);
      break;
    case 1:
      v19 = sub_1DD5A70(a3);
      v15 = a1[32];
      v16 = v19;
      v17 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v15 + 480LL);
      goto LABEL_7;
    case 2:
      v14 = sub_1DD5A70(a3);
      v15 = a1[32];
      v16 = v14;
      v17 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v15 + 488LL);
LABEL_7:
      v18 = sub_38CF310(v16, 0, a1[31], 0);
      return v17(v15, v18);
    case 3:
      v20 = a1[31];
      if ( *(_BYTE *)(a1[30] + 296LL) )
      {
        v21 = sub_396FFC0((__int64)a1, a4, *(_DWORD *)(a3 + 48));
        v9 = (unsigned int *)sub_38CF310(v21, 0, v20, 0);
      }
      else
      {
        v22 = sub_1DD5A70(a3);
        v23 = sub_38CF310(v22, 0, v20, 0);
        v24 = *(__int64 (**)())(**(_QWORD **)(a1[33] + 16LL) + 56LL);
        if ( v24 == sub_1D12D20 )
LABEL_14:
          BUG();
        v25 = v24();
        v26 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v25 + 1040LL))(
                v25,
                a1[33],
                a4,
                a1[31]);
        v9 = (unsigned int *)sub_38CB1F0(17, v23, v26, a1[31], 0);
      }
      break;
    case 5:
      v7 = *(__int64 (**)())(**(_QWORD **)(a1[33] + 16LL) + 56LL);
      if ( v7 == sub_1D12D20 )
        goto LABEL_14;
      v8 = v7();
      v9 = (unsigned int *)(*(__int64 (__fastcall **)(__int64, _DWORD *, __int64, _QWORD, _QWORD))(*(_QWORD *)v8 + 1024LL))(
                             v8,
                             a2,
                             a3,
                             a4,
                             a1[31]);
      break;
    default:
      v9 = 0;
      break;
  }
  v12 = sub_396DDB0((__int64)a1);
  sub_1E0A730(a2, v12);
  return sub_38DDD30(a1[32], v9);
}

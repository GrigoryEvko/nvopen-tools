// Function: sub_A530C0
// Address: 0xa530c0
//
__int64 __fastcall sub_A530C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  _BYTE *v4; // rax
  __int64 v5; // r14
  unsigned __int8 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rdx
  unsigned int v10; // ebx
  int v11; // r14d
  _BYTE *v12; // rax
  __int64 v13; // r15
  __int64 v14; // rdi
  _WORD *v15; // rdx
  __int64 v17; // rdi
  _WORD *v18; // rdx
  int v19; // eax
  __int64 v20; // rdi
  int v21; // r15d
  _BYTE *v22; // rax
  _QWORD *v23; // rdx

  v3 = *(_QWORD *)a1;
  v4 = *(_BYTE **)(v3 + 32);
  if ( (unsigned __int64)v4 >= *(_QWORD *)(v3 + 24) )
  {
    sub_CB5D20(v3, 33);
  }
  else
  {
    *(_QWORD *)(v3 + 32) = v4 + 1;
    *v4 = 33;
  }
  v5 = *(_QWORD *)a1;
  v6 = (unsigned __int8 *)sub_B91B20(a2);
  sub_A518E0(v6, v7, v5);
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v9) <= 4 )
  {
    sub_CB6200(v8, " = !{", 5);
  }
  else
  {
    *(_DWORD *)v9 = 555760928;
    *(_BYTE *)(v9 + 4) = 123;
    *(_QWORD *)(v8 + 32) += 5LL;
  }
  v10 = 0;
  v11 = sub_B91A00(a2);
  if ( v11 )
  {
    while ( 1 )
    {
      v12 = (_BYTE *)sub_B91A10(a2, v10);
      v13 = (__int64)v12;
      if ( *v12 == 7 )
      {
        if ( !byte_4F80910 && (unsigned int)sub_2207590(&byte_4F80910) )
        {
          qword_4F80928 = 0;
          qword_4F80920 = (__int64)off_4979428;
          qword_4F80930 = 0;
          qword_4F80938 = 0;
          __cxa_atexit(nullsub_37, &qword_4F80920, &qword_4A427C0);
          sub_2207640(&byte_4F80910);
        }
        sub_A52D40(*(_QWORD *)a1, v13);
      }
      else
      {
        v19 = (*(__int64 (__fastcall **)(_QWORD, _BYTE *))(**(_QWORD **)(a1 + 32) + 32LL))(*(_QWORD *)(a1 + 32), v12);
        v20 = *(_QWORD *)a1;
        v21 = v19;
        if ( v19 == -1 )
        {
          v23 = *(_QWORD **)(v20 + 32);
          if ( *(_QWORD *)(v20 + 24) - (_QWORD)v23 <= 7u )
          {
            sub_CB6200(v20, "<badref>", 8);
          }
          else
          {
            *v23 = 0x3E6665726461623CLL;
            *(_QWORD *)(v20 + 32) += 8LL;
          }
        }
        else
        {
          v22 = *(_BYTE **)(v20 + 32);
          if ( (unsigned __int64)v22 >= *(_QWORD *)(v20 + 24) )
          {
            v20 = sub_CB5D20(v20, 33);
          }
          else
          {
            *(_QWORD *)(v20 + 32) = v22 + 1;
            *v22 = 33;
          }
          sub_CB59F0(v20, v21);
        }
      }
      if ( v11 == ++v10 )
        break;
      v17 = *(_QWORD *)a1;
      v18 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
      if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v18 <= 1u )
      {
        sub_CB6200(v17, ", ", 2);
      }
      else
      {
        *v18 = 8236;
        *(_QWORD *)(v17 + 32) += 2LL;
      }
    }
  }
  v14 = *(_QWORD *)a1;
  v15 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
  if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v15 <= 1u )
    return sub_CB6200(v14, &unk_43260C9, 2);
  *v15 = 2685;
  *(_QWORD *)(v14 + 32) += 2LL;
  return 2685;
}

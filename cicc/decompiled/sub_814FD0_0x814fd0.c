// Function: sub_814FD0
// Address: 0x814fd0
//
__int64 __fastcall sub_814FD0(__int64 a1, _QWORD *a2)
{
  char v4; // r13
  _QWORD *v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rdi
  __int64 result; // rax
  __int64 v9; // r14
  int v10; // r13d
  unsigned int v11; // esi
  __int64 v12; // r13
  int v13[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = *(_BYTE *)(a1 + 176);
  if ( !(unsigned int)sub_8D32E0(*(_QWORD *)(a1 + 128)) )
  {
    *a2 += 2LL;
    sub_8238B0(qword_4F18BE0, "ad", 2);
  }
  v5 = (_QWORD *)qword_4F18BE0;
  ++*a2;
  v6 = v5[2];
  if ( (unsigned __int64)(v6 + 1) > v5[1] )
  {
    sub_823810(v5);
    v5 = (_QWORD *)qword_4F18BE0;
    v6 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v5[4] + v6) = 76;
  ++v5[2];
  if ( v4 == 1 )
  {
    v12 = *(_QWORD *)(a1 + 184);
    sub_80BBA0(*(_QWORD *)(a1 + 128), a2);
    sub_812EE0(v12, a2);
    goto LABEL_9;
  }
  if ( !v4 )
  {
    v9 = *(_QWORD *)(a1 + 184);
    v13[0] = 0;
    if ( sub_80A070(v9, v13) )
    {
      v10 = dword_4D0425C;
      if ( dword_4D0425C )
      {
        v10 = v13[0];
        if ( v13[0] )
        {
          v11 = v13[0];
          v10 = 0;
          goto LABEL_16;
        }
      }
    }
    else
    {
      v13[0] = 1;
      v11 = 1;
      v10 = 1;
      if ( dword_4D0425C )
        goto LABEL_16;
    }
    sub_80BBA0(*(_QWORD *)(a1 + 128), a2);
    v11 = v13[0];
LABEL_16:
    sub_8111C0(v9, v11, v10, 1, 0, 0, (__int64)a2);
    goto LABEL_9;
  }
  if ( v4 != 5 )
    sub_721090();
  *a2 += 2LL;
  sub_8238B0(v5, &unk_3C1BC40, 2);
  *a2 += 2LL;
  sub_8238B0(qword_4F18BE0, &unk_42EACF9, 2);
  sub_80F5E0(*(_QWORD *)(a1 + 184), 0, a2);
LABEL_9:
  v7 = (_QWORD *)qword_4F18BE0;
  ++*a2;
  result = v7[2];
  if ( (unsigned __int64)(result + 1) > v7[1] )
  {
    sub_823810(v7);
    v7 = (_QWORD *)qword_4F18BE0;
    result = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v7[4] + result) = 69;
  ++v7[2];
  return result;
}

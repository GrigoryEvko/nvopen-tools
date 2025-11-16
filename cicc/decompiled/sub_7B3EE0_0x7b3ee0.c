// Function: sub_7B3EE0
// Address: 0x7b3ee0
//
__int64 __fastcall sub_7B3EE0(unsigned __int8 *a1, _QWORD *a2)
{
  unsigned __int8 *v2; // r12
  unsigned __int64 v3; // rbx
  __int64 v4; // rdi
  __int64 v5; // r12
  char v6; // r8
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  unsigned __int8 *v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v14; // rax
  int v15; // [rsp+Ch] [rbp-34h] BYREF
  unsigned __int8 *v16; // [rsp+10h] [rbp-30h] BYREF
  unsigned __int64 v17[5]; // [rsp+18h] [rbp-28h] BYREF

  v2 = a1;
  v3 = (unsigned __int64)&a1[*a2 - 1];
  v4 = qword_4F17F68;
  if ( !qword_4F17F68 )
  {
    qword_4F17F68 = sub_8237A0(128);
    v4 = qword_4F17F68;
  }
  sub_823800(v4);
  v16 = v2;
  while ( v3 >= (unsigned __int64)v2 )
  {
    while ( *v2 == 92 )
    {
      if ( (v2[1] & 0xDF) != 0x55 )
        goto LABEL_16;
      v14 = sub_7B39D0((unsigned __int64 *)&v16, 0, 0, 0);
      sub_7AD610(v14);
      v2 = v16;
LABEL_13:
      if ( v3 < (unsigned __int64)v2 )
        goto LABEL_19;
    }
    if ( (*v2 & 0x80u) != 0 )
    {
      v5 = (int)sub_722680(v2, v17, &v15, unk_4F064A8 == 0);
      if ( v15 )
      {
        sub_7B0EB0((unsigned __int64)v16, (__int64)dword_4F07508);
        sub_684AC0(8u, 0x6BCu);
      }
      v6 = v17[0];
      if ( v17[0] > 0x7F )
      {
        sub_7AD610(v17[0]);
      }
      else
      {
        v7 = qword_4F17F68;
        v8 = *(_QWORD *)(qword_4F17F68 + 16);
        if ( (unsigned __int64)(v8 + 1) > *(_QWORD *)(qword_4F17F68 + 8) )
        {
          sub_823810(qword_4F17F68);
          v7 = qword_4F17F68;
          v6 = v17[0];
          v8 = *(_QWORD *)(qword_4F17F68 + 16);
        }
        *(_BYTE *)(*(_QWORD *)(v7 + 32) + v8) = v6;
        ++*(_QWORD *)(v7 + 16);
      }
      v2 = &v16[v5];
      v16 = v2;
      goto LABEL_13;
    }
LABEL_16:
    v9 = qword_4F17F68;
    v10 = *(_QWORD *)(qword_4F17F68 + 16);
    if ( (unsigned __int64)(v10 + 1) > *(_QWORD *)(qword_4F17F68 + 8) )
    {
      sub_823810(qword_4F17F68);
      v9 = qword_4F17F68;
      v10 = *(_QWORD *)(qword_4F17F68 + 16);
    }
    v11 = v16++;
    *(_BYTE *)(*(_QWORD *)(v9 + 32) + v10) = *v11;
    v2 = v16;
    ++*(_QWORD *)(v9 + 16);
  }
LABEL_19:
  v12 = qword_4F17F68;
  *a2 = *(_QWORD *)(qword_4F17F68 + 16);
  return *(_QWORD *)(v12 + 32);
}

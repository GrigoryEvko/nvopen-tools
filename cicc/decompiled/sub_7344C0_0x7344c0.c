// Function: sub_7344C0
// Address: 0x7344c0
//
void __fastcall sub_7344C0(__int64 a1, int a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 v6; // rbx
  _BYTE *v7; // rdx
  __int64 v8; // rax
  bool v9; // zf

  v3 = qword_4F04C68[0];
  v4 = qword_4F04C68[0] + 776LL * a2;
  if ( !dword_4F07590 )
  {
    if ( (*(_BYTE *)(v4 + 6) & 2) != 0 )
    {
LABEL_3:
      v4 = v3;
      goto LABEL_4;
    }
    if ( sub_734480(a1) )
    {
      v3 = qword_4F04C68[0];
      goto LABEL_3;
    }
  }
LABEL_4:
  v5 = v4;
  v6 = v4 + 32;
  v7 = sub_732EF0(v5);
  v8 = *(_QWORD *)(v6 - 8);
  if ( !v8 )
    v8 = v6;
  if ( *((_QWORD *)v7 + 34) )
    *(_QWORD *)(*(_QWORD *)(v8 + 104) + 112LL) = a1;
  else
    *((_QWORD *)v7 + 34) = a1;
  *(_QWORD *)(v8 + 104) = a1;
  v9 = *(_QWORD *)(a1 + 40) == 0;
  *(_QWORD *)(a1 + 112) = 0;
  if ( v9 && (*(_BYTE *)(a1 + 89) & 2) == 0 )
    sub_72EE40(a1, 0x3Bu, (__int64)v7);
}

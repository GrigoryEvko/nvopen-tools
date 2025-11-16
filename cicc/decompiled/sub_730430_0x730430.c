// Function: sub_730430
// Address: 0x730430
//
void __fastcall sub_730430(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rax

  if ( (*(_BYTE *)(a1 + 120) & 0x20) != 0 )
    v1 = *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 184);
  else
    v1 = qword_4F04C50;
  v2 = *(int *)(v1 + 240);
  v3 = *(_QWORD *)(v1 + 136);
  if ( (_DWORD)v2 == -1 )
  {
    *(_QWORD *)(a1 + 112) = v3;
    *(_QWORD *)(v1 + 136) = a1;
    sub_72EE40(a1, 0xCu, v1);
  }
  else
  {
    v4 = qword_4F04C68[0] + 776 * v2;
    if ( v3 )
      *(_QWORD *)(*(_QWORD *)(v4 + 296) + 112LL) = a1;
    else
      *(_QWORD *)(v1 + 136) = a1;
    *(_QWORD *)(v4 + 296) = a1;
    *(_QWORD *)(a1 + 112) = 0;
    sub_72EE40(a1, 0xCu, v1);
  }
}

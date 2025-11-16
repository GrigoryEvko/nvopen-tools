// Function: sub_7E13F0
// Address: 0x7e13f0
//
void __fastcall sub_7E13F0(__int64 a1, __int64 *a2, _QWORD *a3, _QWORD *a4, __int64 *a5)
{
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax

  v8 = *(_QWORD *)(a1 + 200);
  if ( v8 )
  {
    v9 = *(_QWORD *)(a1 + 176);
    v10 = 0;
    if ( v9 )
    {
      v10 = *(_QWORD *)(v9 + 104);
      if ( (*(_BYTE *)(a1 + 192) & 1) != 0 )
        v10 = -v10;
    }
    *a2 = v10;
    if ( unk_4F0687C )
      *a2 = ((*(_BYTE *)(v8 + 192) & 2) != 0) + 2 * v10;
    *a3 = 0;
    *a5 = 0;
    if ( (*(_BYTE *)(v8 + 192) & 2) != 0 )
    {
      v11 = *(unsigned __int16 *)(v8 + 224);
      v12 = v11 * sub_7E1340();
      *a5 = v12;
      if ( !unk_4F0687C && (*(_BYTE *)(v8 + 192) & 2) != 0 )
        *a5 = v12 | 1;
      *a4 = 0;
    }
    else
    {
      *a4 = v8;
    }
  }
  else
  {
    *a2 = 0;
    *a3 = 0;
    *a5 = 0;
    *a4 = 0;
  }
}

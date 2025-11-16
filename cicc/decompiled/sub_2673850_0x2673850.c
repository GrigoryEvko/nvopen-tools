// Function: sub_2673850
// Address: 0x2673850
//
__int64 __fastcall sub_2673850(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // r15
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // rdx
  _BYTE *v14; // rax

  v3 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v3 == 85 && a2 == v3 - 32 )
  {
    v5 = *a1;
    if ( *(char *)(v3 + 7) >= 0 )
      goto LABEL_14;
    v7 = sub_BD2BC0(*(_QWORD *)(a2 + 24));
    v9 = v7 + v8;
    v10 = 0;
    if ( *(char *)(v3 + 7) < 0 )
      v10 = sub_BD2BC0(v3);
    if ( !(unsigned int)((v9 - v10) >> 4) )
    {
LABEL_14:
      if ( (!v5
         || (v11 = *(_QWORD *)(v5 + 120)) != 0
         && (v12 = *(_QWORD *)(v3 - 32)) != 0
         && !*(_BYTE *)v12
         && *(_QWORD *)(v12 + 24) == *(_QWORD *)(v3 + 80)
         && v11 == v12)
        && a3 == a1[1] )
      {
        v13 = (_QWORD *)a1[2];
        v14 = *(_BYTE **)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
        if ( v14 != (_BYTE *)*v13 )
        {
          if ( *v14 > 3u )
            v14 = 0;
          else
            *(_BYTE *)a1[3] = *v13 == 0;
        }
        *v13 = v14;
      }
    }
  }
  return 0;
}

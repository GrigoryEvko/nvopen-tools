// Function: sub_35D3D40
// Address: 0x35d3d40
//
bool __fastcall sub_35D3D40(__int64 a1, __int64 a2)
{
  __int64 i; // r14
  __int64 v3; // rcx
  __int64 v4; // r12
  __int64 v5; // r15
  int v6; // r13d
  __int64 *v7; // rax
  char v8; // dl
  int v9; // edx
  __int64 v11; // [rsp+10h] [rbp-60h]
  int v12; // [rsp+1Ch] [rbp-54h]
  __int64 v13; // [rsp+20h] [rbp-50h]
  __int64 v14; // [rsp+28h] [rbp-48h]

  v11 = *(_QWORD *)(a2 + 64);
  v13 = *(_QWORD *)(a2 + 328);
  if ( v13 == a2 + 320 )
    return 0;
  v12 = 0;
  do
  {
    for ( i = *(_QWORD *)(v13 + 56); v13 + 48 != i; i = *(_QWORD *)(i + 8) )
    {
      v3 = *(_QWORD *)(i + 32);
      v4 = v3 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
      if ( v3 != v4 )
      {
        v5 = *(_QWORD *)(i + 32);
        do
        {
          while ( 1 )
          {
            if ( *(_BYTE *)v5 == 8 )
            {
              v6 = *(_DWORD *)(v5 + 24);
              if ( v6 != -1 )
                break;
            }
            v5 += 40;
            if ( v4 == v5 )
              goto LABEL_12;
          }
          v14 = *(_QWORD *)(a1 + 216);
          v7 = sub_2E39F50(*(__int64 **)(a1 + 208), v13);
          if ( v8 && sub_D84450(v14, (unsigned __int64)v7) )
            v9 = 1;
          else
            v9 = 2;
          v12 -= ((unsigned __int8)sub_2E79BA0(v11, v6, v9) == 0) - 1;
          v5 += 40;
        }
        while ( v4 != v5 );
      }
LABEL_12:
      if ( (*(_BYTE *)i & 4) == 0 )
      {
        while ( (*(_BYTE *)(i + 44) & 8) != 0 )
          i = *(_QWORD *)(i + 8);
      }
    }
    v13 = *(_QWORD *)(v13 + 8);
  }
  while ( a2 + 320 != v13 );
  return v12 > 0;
}

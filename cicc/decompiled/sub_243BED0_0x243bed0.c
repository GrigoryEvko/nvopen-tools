// Function: sub_243BED0
// Address: 0x243bed0
//
bool __fastcall sub_243BED0(__int64 a1, int a2, __int64 a3, __int64 *a4)
{
  _DWORD *v5; // rax
  __int64 v6; // r15
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rdx
  bool result; // al
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 i; // r13
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // [rsp+0h] [rbp-70h]
  __int64 v20; // [rsp+18h] [rbp-58h]
  unsigned __int64 v21; // [rsp+20h] [rbp-50h] BYREF
  __int64 v22; // [rsp+28h] [rbp-48h]
  char v23; // [rsp+30h] [rbp-40h]

  if ( !a3 || !*(_QWORD *)(a1 + 8) )
    return 0;
  sub_B2EE70((__int64)&v21, a3, 0);
  if ( !v23 || !(result = sub_D85370(a1, a2, v21)) )
  {
    v5 = *(_DWORD **)(a1 + 8);
    v20 = a3 + 72;
    if ( !v5 || *v5 != 2 )
      goto LABEL_6;
    v17 = 0;
    v11 = *(_QWORD *)(a3 + 80);
    if ( v11 != a3 + 72 )
    {
      do
      {
        if ( !v11 )
          BUG();
        v12 = *(_QWORD *)(v11 + 32);
        for ( i = v11 + 24; i != v12; v17 += v21 )
        {
          while ( 1 )
          {
            if ( !v12 )
              BUG();
            v14 = *(_BYTE *)(v12 - 24);
            if ( v14 == 34 || v14 == 85 )
            {
              v15 = sub_D84370(a1, v12 - 24, 0, 0);
              v22 = v16;
              v21 = v15;
              if ( (_BYTE)v16 )
                break;
            }
            v12 = *(_QWORD *)(v12 + 8);
            if ( i == v12 )
              goto LABEL_27;
          }
          v12 = *(_QWORD *)(v12 + 8);
        }
LABEL_27:
        v11 = *(_QWORD *)(v11 + 8);
      }
      while ( v11 != v20 );
    }
    result = sub_D85370(a1, a2, v17);
    if ( !result )
    {
LABEL_6:
      v6 = *(_QWORD *)(a3 + 80);
      if ( v6 != v20 )
      {
        while ( 1 )
        {
          v7 = v6 - 24;
          if ( !v6 )
            v7 = 0;
          v8 = sub_FDD2C0(a4, v7, 0);
          v22 = v9;
          v21 = v8;
          if ( (_BYTE)v9 )
          {
            result = sub_D85370(a1, a2, v21);
            if ( result )
              break;
          }
          v6 = *(_QWORD *)(v6 + 8);
          if ( v6 == v20 )
            return 0;
        }
        return result;
      }
      return 0;
    }
  }
  return result;
}

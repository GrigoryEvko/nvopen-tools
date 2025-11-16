// Function: sub_89A200
// Address: 0x89a200
//
__int64 __fastcall sub_89A200(__int64 *a1, _QWORD *a2, int *a3)
{
  __int64 **v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // rdi
  __int64 result; // rax
  __int64 **v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 *v19; // rdi
  int v20; // edx
  __int64 v21; // rdx
  __int64 *v22; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v23[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( !dword_4F07588
    || dword_4F04C64 == -1
    || dword_4F04C44 != -1
    || (v21 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v21 + 6) & 6) != 0)
    || (result = 0, *(_BYTE *)(v21 + 4) == 12) )
  {
    v7 = (__int64 **)a1;
    v23[0] = 0;
    sub_89A1A0(a2, a1, v23, &v22);
    v12 = v22;
    if ( v22 )
    {
      while ( 1 )
      {
        if ( !a2 || (*((_BYTE *)v23[0] + 57) & 4) != 0 )
        {
          result = sub_88D7A0((__int64)v12, (__int64)v7, v8, v9, v10, v11);
          if ( (_DWORD)result )
            break;
        }
        v7 = &v22;
        sub_89A1C0((__int64 *)v23, &v22);
        v12 = v22;
        if ( !v22 )
          goto LABEL_9;
      }
      if ( a3 )
      {
        v20 = 1;
LABEL_18:
        *a3 = v20;
      }
    }
    else
    {
LABEL_9:
      result = 0;
      if ( a3 )
      {
        v14 = (__int64 **)a1;
        sub_89A1A0(a2, a1, v23, &v22);
        v19 = v22;
        if ( v22 )
        {
          while ( 1 )
          {
            if ( (*((_BYTE *)v23[0] + 57) & 4) == 0 )
            {
              v20 = sub_88D7A0((__int64)v19, (__int64)v14, v15, v16, v17, v18);
              if ( v20 )
                break;
            }
            v14 = &v22;
            sub_89A1C0((__int64 *)v23, &v22);
            v19 = v22;
            if ( !v22 )
              goto LABEL_15;
          }
          result = 0;
        }
        else
        {
LABEL_15:
          result = 0;
          v20 = 0;
        }
        goto LABEL_18;
      }
    }
  }
  return result;
}

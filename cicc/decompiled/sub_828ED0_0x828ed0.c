// Function: sub_828ED0
// Address: 0x828ed0
//
__int64 __fastcall sub_828ED0(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        int a4,
        int a5,
        char a6,
        int a7,
        _DWORD *a8,
        _DWORD *a9)
{
  char v9; // bl
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 result; // rax
  int v13; // eax
  __int64 v14; // r8
  unsigned int v15; // r12d
  unsigned int v16; // eax
  unsigned int v19; // [rsp+10h] [rbp-40h]

  if ( a8 )
    *a8 = 0;
  if ( a9 )
    *a9 = 0;
  if ( (*(_DWORD *)(a1 + 80) & 0x40001000) == 0x40000000 )
    return 0;
  v9 = *(_BYTE *)(a1 + 80);
  if ( v9 == 16 )
  {
    v10 = *(__int64 **)(a1 + 88);
    a1 = *v10;
    v9 = *(_BYTE *)(*v10 + 80);
  }
  if ( v9 == 24 )
  {
    a1 = *(_QWORD *)(a1 + 88);
    v9 = *(_BYTE *)(a1 + 80);
  }
  if ( dword_4D047C8
    && (!a4
     || dword_4D047AC && (dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0))
    && unk_4F04C48 != -1
    && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0
    && (*(_BYTE *)(a1 + 81) & 0x10) == 0 )
  {
    v19 = a3;
    v13 = sub_8809D0(a1);
    a3 = v19;
    if ( !v13 )
    {
      v14 = *(unsigned int *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 200);
      if ( (int)v14 <= 0 && unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
      {
        if ( dword_4D047C8 )
        {
          v15 = *(_DWORD *)(a1 + 44);
          v16 = sub_7D3BE0(a1, a2, v19, dword_4D047C8, v14);
          a3 = v19;
          if ( v16 )
          {
            if ( v15 > v16 )
            {
              if ( a5 )
              {
                if ( !dword_4F077BC )
                {
LABEL_34:
                  if ( !a7 )
                  {
                    if ( a9 )
                    {
                      *a9 = 1;
                      return 0;
                    }
                    return 0;
                  }
                  goto LABEL_11;
                }
                if ( (unsigned __int64)(qword_4F077A8 - 30400LL) <= 0x25E3 )
                  goto LABEL_11;
              }
              if ( dword_4F077BC && (a6 & 1) != 0 )
                goto LABEL_11;
              goto LABEL_34;
            }
          }
        }
      }
    }
  }
LABEL_11:
  v11 = *(_QWORD *)(a1 + 88);
  if ( v9 != 20 )
  {
    if ( a2 )
      return 0;
    result = 1;
    if ( !a3 )
      return result;
LABEL_14:
    if ( *(char *)(v11 + 193) >= 0 )
      return result;
    if ( a8 )
      *a8 = 1;
    return 0;
  }
  v11 = *(_QWORD *)(v11 + 176);
  result = 1;
  if ( a3 )
    goto LABEL_14;
  return result;
}

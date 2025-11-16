// Function: sub_734F30
// Address: 0x734f30
//
__int64 sub_734F30()
{
  __int64 result; // rax
  __int64 v1; // r12
  int i; // ebx
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rdi

  result = dword_4F073A0;
  if ( (int)dword_4F073A0 > 0 )
  {
    v1 = 16;
    for ( i = 1; (int)dword_4F073A0 >= i; ++i )
    {
      result = v1 + unk_4F072B8;
      v4 = *(int *)(v1 + unk_4F072B8 + 8);
      if ( *(_QWORD *)(unk_4F073B0 + 8 * v4) )
      {
        v5 = *(_QWORD *)result;
        if ( *(_QWORD *)result )
        {
          result = unk_4D03FF0;
          if ( (_QWORD *)unk_4D03FF0 == qword_4D03FD0 )
          {
            if ( (*(_BYTE *)(v5 - 8) & 2) == 0 )
            {
              v3 = *(_QWORD *)(v5 + 32);
              if ( (*(_BYTE *)(v3 + 208) & 1) == 0 )
              {
LABEL_15:
                result = *(_BYTE *)(*(_QWORD *)(unk_4F072B0 + 8 * v4) + 29LL) & 1;
LABEL_6:
                if ( (*(_BYTE *)(v3 + 203) & 8) == 0 )
                {
                  if ( (_DWORD)result )
                    result = sub_734690((_QWORD *)v5);
                }
                goto LABEL_9;
              }
LABEL_5:
              result = *(_BYTE *)(v5 + 29) & 1;
              goto LABEL_6;
            }
          }
          else if ( unk_4D03FF0 == *(_QWORD *)(unk_4F066A0 + 8LL * *(int *)(v5 + 24)) )
          {
            v3 = *(_QWORD *)(v5 + 32);
            if ( (*(_BYTE *)(v3 + 208) & 1) == 0 )
              goto LABEL_15;
            goto LABEL_5;
          }
        }
      }
LABEL_9:
      v1 += 16;
    }
  }
  return result;
}

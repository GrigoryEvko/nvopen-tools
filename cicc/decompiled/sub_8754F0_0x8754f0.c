// Function: sub_8754F0
// Address: 0x8754f0
//
int __fastcall sub_8754F0(__int16 a1, unsigned __int8 *a2, __int64 a3)
{
  int result; // eax
  unsigned int v5; // r14d
  unsigned int v6; // r14d
  __int64 v7; // r12
  int v8; // r13d
  const char *v9; // rax
  unsigned int v10; // [rsp+8h] [rbp-38h] BYREF
  int v11; // [rsp+Ch] [rbp-34h] BYREF
  char *v12; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v13[5]; // [rsp+18h] [rbp-28h] BYREF

  if ( !dword_4F5FD98 )
  {
    sub_7461E0((__int64)&qword_4F5FDA0);
    byte_4F5FE31 = 0;
    qword_4F5FDA0 = (__int64)sub_8754E0;
    byte_4F5FE35 = 1;
    dword_4F5FD98 = 1;
  }
  result = a2[80] - 14;
  if ( (unsigned __int8)(a2[80] - 14) > 1u )
  {
    result = sub_879510(a2);
    if ( !result )
    {
      if ( *(_DWORD *)a3 )
      {
        if ( (a1 & 0x8001) != 0 )
        {
          v5 = (a1 & 0x8000) == 0 ? 0xFFFFFFF0 : 0;
          if ( (a1 & 2) != 0 )
            v6 = v5 + 84;
          else
            v6 = v5 + 116;
        }
        else
        {
          v6 = 32;
          if ( (a1 & 4) != 0 )
          {
            if ( (a1 & 8) != 0 )
            {
              v6 = (a1 & 0x10) == 0 ? 85 : 67;
            }
            else
            {
              v6 = 77;
              if ( (a1 & 0x10) == 0 )
              {
                v6 = 65;
                if ( (a1 & 0x20) == 0 )
                  v6 = (a1 & 0x40) == 0 ? 82 : 69;
              }
            }
          }
        }
        sub_729E00(*(_DWORD *)a3, &v12, v13, &v10, &v11);
        fprintf(qword_4D04900, "%p\t", a2);
        sub_87D380(a2, &qword_4F5FDA0);
        v7 = v10;
        v8 = *(unsigned __int16 *)(a3 + 4);
        v9 = (const char *)sub_723260(v12);
        return fprintf(qword_4D04900, "\t%c\t%s\t%lu\t%d\n", v6, v9, v7, v8);
      }
    }
  }
  return result;
}

// Function: sub_ED1F20
// Address: 0xed1f20
//
__int64 *__fastcall sub_ED1F20(__int64 *a1, unsigned int *a2)
{
  unsigned int v3; // r10d
  signed __int64 v4; // r11
  _DWORD *v6; // rdi
  int v7; // r8d
  int v8; // esi
  unsigned __int8 *v9; // rax
  int v10; // edx
  int v11; // ecx
  const char *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rbx
  const char *v16; // [rsp+0h] [rbp-40h] BYREF
  char v17; // [rsp+20h] [rbp-20h]
  char v18; // [rsp+21h] [rbp-1Fh]

  v3 = a2[1];
  if ( v3 > 3 )
  {
    sub_ED0840(a1, 9, "number of value profile kinds is invalid");
    return a1;
  }
  else
  {
    v4 = *a2;
    if ( (v4 & 7) != 0 )
    {
      sub_ED0840(a1, 9, "total size is not multiples of quardword");
      return a1;
    }
    else
    {
      v6 = a2 + 2;
      if ( v3 )
      {
        v7 = 0;
        while ( *v6 <= 2u )
        {
          v8 = v6[1];
          if ( v8 )
          {
            v9 = (unsigned __int8 *)(v6 + 2);
            v10 = 0;
            do
            {
              v11 = *v9++;
              v10 += v11;
            }
            while ( (unsigned __int8 *)((char *)v6 + (unsigned int)(v8 - 1) + 9) != v9 );
            v6 = (_DWORD *)((char *)v6 + 16 * v10 + ((v8 + 15) & 0xFFFFFFF8));
            if ( (char *)v6 - (char *)a2 > v4 )
            {
LABEL_14:
              v18 = 1;
              v13 = "value profile address is greater than total size";
              goto LABEL_17;
            }
          }
          else
          {
            v6 += 2;
            if ( (char *)v6 - (char *)a2 > v4 )
              goto LABEL_14;
          }
          if ( v3 == ++v7 )
            goto LABEL_11;
        }
        v18 = 1;
        v13 = "value kind is invalid";
LABEL_17:
        v16 = v13;
        v17 = 3;
        v14 = sub_22077B0(48);
        v15 = v14;
        if ( v14 )
        {
          *(_DWORD *)(v14 + 8) = 9;
          *(_QWORD *)v14 = &unk_49E4BC8;
          sub_CA0F50((__int64 *)(v14 + 16), (void **)&v16);
        }
        *a1 = v15 | 1;
        return a1;
      }
      else
      {
LABEL_11:
        *a1 = 1;
        return a1;
      }
    }
  }
}

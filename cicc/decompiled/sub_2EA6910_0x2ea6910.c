// Function: sub_2EA6910
// Address: 0x2ea6910
//
__int64 __fastcall sub_2EA6910(__int64 a1, unsigned int a2)
{
  __int64 v3; // r12
  __int64 v4; // r15
  unsigned int v5; // r13d
  __int64 v6; // rdi
  __int64 (*v7)(); // rax
  unsigned int v9; // r12d
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // rcx
  __int64 v15; // rax

  v3 = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 32LL);
  v4 = *(_QWORD *)(v3 + 32);
  v5 = sub_2EBF3A0(v4, a2);
  if ( !(_BYTE)v5 )
  {
    v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v3 + 16) + 200LL))(*(_QWORD *)(v3 + 16));
    v7 = *(__int64 (**)())(*(_QWORD *)v6 + 192LL);
    if ( v7 != sub_2EA3FC0 )
    {
      v9 = ((__int64 (__fastcall *)(__int64, _QWORD))v7)(v6, a2);
      if ( (_BYTE)v9 )
      {
        if ( (a2 & 0x80000000) != 0 )
          v10 = *(_QWORD *)(*(_QWORD *)(v4 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
        else
          v10 = *(_QWORD *)(*(_QWORD *)(v4 + 304) + 8LL * a2);
        if ( v10 )
        {
          if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0
            || (v10 = *(_QWORD *)(v10 + 32)) != 0 && (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
          {
            v11 = *(_QWORD *)(v10 + 16);
LABEL_10:
            v12 = *(_QWORD *)(v11 + 24);
            if ( *(_BYTE *)(a1 + 84) )
            {
              v13 = *(_QWORD **)(a1 + 64);
              v14 = &v13[*(unsigned int *)(a1 + 76)];
              if ( v13 == v14 )
              {
LABEL_17:
                v15 = v11;
LABEL_19:
                while ( 1 )
                {
                  v10 = *(_QWORD *)(v10 + 32);
                  if ( !v10 || (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
                    return v9;
                  v11 = *(_QWORD *)(v10 + 16);
                  if ( v11 != v15 )
                    goto LABEL_10;
                }
              }
              while ( v12 != *v13 )
              {
                if ( v14 == ++v13 )
                  goto LABEL_17;
              }
            }
            else if ( !sub_C8CA60(a1 + 56, v12) )
            {
              v15 = *(_QWORD *)(v10 + 16);
              goto LABEL_19;
            }
            return 0;
          }
        }
        return v9;
      }
    }
  }
  return v5;
}

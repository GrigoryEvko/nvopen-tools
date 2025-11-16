// Function: sub_E352E0
// Address: 0xe352e0
//
__int64 __fastcall sub_E352E0(__int64 a1, __int64 a2)
{
  int v2; // ebx
  _BYTE *v3; // rax
  _BYTE *v4; // r14
  __int64 result; // rax
  const char *v6; // rax
  _BYTE v7[16]; // [rsp+0h] [rbp-70h] BYREF
  __int64 (__fastcall *v8)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-60h]
  const char *v9; // [rsp+20h] [rbp-50h] BYREF
  char v10; // [rsp+40h] [rbp-30h]
  char v11; // [rsp+41h] [rbp-2Fh]

  v2 = sub_E345B0((_BYTE *)a2);
  v3 = sub_E34DB0(a1, (unsigned __int8 *)a2);
  v4 = v3;
  if ( v2 == 1 )
  {
    if ( !(unsigned __int8)sub_E34600(a2) )
    {
      sub_E45390(v7, a1 + 144, a2);
      v11 = 1;
      v6 = "Entry intrinsic can occur only in a convergent function.";
      goto LABEL_25;
    }
    if ( sub_AA5B70(*(_QWORD *)(a2 + 40)) )
    {
      if ( !*(_BYTE *)(a1 + 192) )
        goto LABEL_15;
      sub_E45390(v7, a1 + 144, a2);
      v11 = 1;
      v6 = "Entry intrinsic cannot be preceded by a convergent operation in the same basic block.";
    }
    else
    {
      sub_E45390(v7, a1 + 144, a2);
      v11 = 1;
      v6 = "Entry intrinsic can occur only in the entry block.";
    }
  }
  else
  {
    if ( v2 != 2 )
    {
      if ( v2 )
      {
        if ( v2 == 3 )
        {
          if ( !(unsigned __int8)sub_E34620((unsigned __int8 *)a2) )
          {
LABEL_6:
            if ( !v4 && v2 == 3 )
            {
              result = sub_E34620((unsigned __int8 *)a2);
              if ( !(_BYTE)result )
                return result;
              result = *(unsigned int *)(a1 + 152);
              if ( (_DWORD)result )
              {
                *(_DWORD *)(a1 + 152) = 1;
                return result;
              }
LABEL_29:
              sub_E45390(v7, a1 + 144, a2);
              v11 = 1;
              v6 = "Cannot mix controlled and uncontrolled convergence in the same function.";
              goto LABEL_25;
            }
LABEL_18:
            result = sub_E34620((unsigned __int8 *)a2);
            if ( !(_BYTE)result )
            {
              sub_E45390(v7, a1 + 144, a2);
              v11 = 1;
              v6 = "Convergence control token can only be used in a convergent call.";
              goto LABEL_25;
            }
            if ( *(_DWORD *)(a1 + 152) != 1 )
            {
              *(_DWORD *)(a1 + 152) = 0;
              return result;
            }
            goto LABEL_29;
          }
LABEL_17:
          *(_BYTE *)(a1 + 192) = 1;
          goto LABEL_6;
        }
LABEL_16:
        nullsub_315();
        if ( !(unsigned __int8)sub_E34620((unsigned __int8 *)a2) )
          goto LABEL_18;
        goto LABEL_17;
      }
LABEL_15:
      if ( v4 )
      {
        sub_E45390(v7, a1 + 144, a2);
        v11 = 1;
        v6 = "Entry or anchor intrinsic cannot have a convergencectrl token operand.";
        goto LABEL_25;
      }
      goto LABEL_16;
    }
    if ( v3 )
    {
      if ( *(_BYTE *)(a1 + 192) )
      {
        sub_E45390(v7, a1 + 144, a2);
        v11 = 1;
        v6 = "Loop intrinsic cannot be preceded by a convergent operation in the same basic block.";
        goto LABEL_25;
      }
      goto LABEL_16;
    }
    sub_E45390(v7, a1 + 144, a2);
    v11 = 1;
    v6 = "Loop intrinsic must have a convergencectrl token operand.";
  }
LABEL_25:
  v9 = v6;
  v10 = 3;
  sub_E348A0((_BYTE *)a1, (__int64)&v9, v7, 1);
  result = (__int64)v8;
  if ( v8 )
    return v8(v7, v7, 3);
  return result;
}

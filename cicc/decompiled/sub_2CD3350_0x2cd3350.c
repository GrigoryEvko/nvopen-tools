// Function: sub_2CD3350
// Address: 0x2cd3350
//
void __fastcall sub_2CD3350(_BYTE *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // eax

  v2 = *(_QWORD *)(a2 - 32);
  if ( v2 && !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
  {
    v3 = *(_DWORD *)(v2 + 36);
    if ( v3 == 8816 )
    {
      sub_2CD2F00(a1, a2, (__int64)"__nvvm_frexpq", 0xDu, 2);
    }
    else if ( v3 > 0x2270 )
    {
      if ( v3 == 9237 )
      {
        sub_2CD2F00(a1, a2, (__int64)"__nvvm_powq", 0xBu, 2);
      }
      else if ( v3 > 0x2415 )
      {
        if ( v3 == 9535 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_sqrtq", 0xCu, 1);
        }
        else if ( v3 <= 0x253F )
        {
          if ( v3 == 9408 )
          {
            sub_2CD2F00(a1, a2, (__int64)"__nvvm_roundq", 0xDu, 1);
          }
          else if ( v3 <= 0x24C0 )
          {
            if ( v3 == 9402 )
            {
              sub_2CD2F00(a1, a2, (__int64)"__nvvm_remainderq", 0x11u, 2);
            }
            else if ( v3 == 9404 )
            {
              sub_2CD2F00(a1, a2, (__int64)"__nvvm_rintq", 0xCu, 1);
            }
          }
          else if ( v3 == 9470 )
          {
            sub_2CD2F00(a1, a2, (__int64)"__nvvm_sinq", 0xBu, 1);
          }
          else if ( v3 == 9471 )
          {
            sub_2CD2F00(a1, a2, (__int64)"__nvvm_sinhq", 0xCu, 1);
          }
        }
        else if ( v3 == 10079 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_tanhq", 0xCu, 1);
        }
        else if ( v3 <= 0x275F )
        {
          if ( v3 == 9577 )
          {
            sub_2CD2F00(a1, a2, (__int64)"__nvvm_subq", 0xBu, 2);
          }
          else if ( v3 == 10077 )
          {
            sub_2CD2F00(a1, a2, (__int64)"__nvvm_tanq", 0xBu, 1);
          }
        }
        else if ( v3 == 10608 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_truncq", 0xDu, 1);
        }
      }
      else if ( v3 == 8990 )
      {
        sub_2CD2F00(a1, a2, (__int64)"__nvvm_logq", 0xBu, 1);
      }
      else if ( v3 <= 0x231E )
      {
        if ( v3 == 8926 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_isnanq", 0xDu, 1);
        }
        else if ( v3 <= 0x22DE )
        {
          if ( v3 == 8863 )
          {
            sub_2CD2F00(a1, a2, (__int64)"__nvvm_hypotq", 0xDu, 2);
          }
          else if ( v3 == 8882 )
          {
            sub_2CD2F00(a1, a2, (__int64)"__nvvm_ilogbq", 0xDu, 1);
          }
        }
        else if ( v3 == 8936 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_isunorderedq", 0x13u, 2);
        }
        else if ( v3 == 8940 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_ldexpq", 0xDu, 2);
        }
      }
      else if ( v3 == 8993 )
      {
        sub_2CD2F00(a1, a2, (__int64)"__nvvm_log2q", 0xCu, 1);
      }
      else if ( v3 <= 0x2321 )
      {
        if ( v3 == 8991 )
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_log10q", 0xDu, 1);
        else
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_log1pq", 0xDu, 1);
      }
      else if ( v3 == 9149 )
      {
        sub_2CD2F00(a1, a2, (__int64)"__nvvm_modfq", 0xCu, 2);
      }
      else if ( v3 == 9155 )
      {
        sub_2CD2F00(a1, a2, (__int64)"__nvvm_mulq", 0xBu, 2);
      }
    }
    else if ( v3 == 8495 )
    {
      sub_2CD2F00(a1, a2, (__int64)"__nvvm_divq", 0xBu, 2);
    }
    else if ( v3 > 0x212F )
    {
      if ( v3 == 8598 )
      {
        sub_2CD2F00(a1, a2, (__int64)"__nvvm_fdimq", 0xCu, 2);
      }
      else if ( v3 <= 0x2196 )
      {
        if ( v3 == 8533 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_exp2q", 0xCu, 1);
        }
        else if ( v3 <= 0x2155 )
        {
          if ( v3 == 8531 )
          {
            sub_2CD2F00(a1, a2, (__int64)"__nvvm_expq", 0xBu, 1);
          }
          else if ( v3 == 8532 )
          {
            sub_2CD2F00(a1, a2, (__int64)"__nvvm_exp10q", 0xDu, 1);
          }
        }
        else if ( v3 == 8535 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_expm1q", 0xDu, 1);
        }
        else if ( v3 == 8591 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_fabsq", 0xCu, 1);
        }
      }
      else if ( v3 == 8712 )
      {
        sub_2CD2F00(a1, a2, (__int64)"__nvvm_fmaxq", 0xCu, 2);
      }
      else if ( v3 <= 0x2208 )
      {
        if ( v3 == 8645 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_floorq", 0xDu, 1);
        }
        else if ( v3 == 8648 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_fmaq", 0xBu, 3);
        }
      }
      else if ( v3 == 8767 )
      {
        sub_2CD2F00(a1, a2, (__int64)"__nvvm_fminq", 0xCu, 2);
      }
      else if ( v3 == 8814 )
      {
        sub_2CD2F00(a1, a2, (__int64)"__nvvm_fmodq", 0xCu, 2);
      }
    }
    else if ( v3 > 0x1FEC )
    {
      if ( v3 == 8311 )
      {
        sub_2CD2F00(a1, a2, (__int64)"__nvvm_cosq", 0xBu, 1);
      }
      else if ( v3 <= 0x2077 )
      {
        if ( v3 == 8291 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_ceilq", 0xCu, 1);
        }
        else if ( v3 == 8305 )
        {
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_copysignq", 0x10u, 2);
        }
      }
      else if ( v3 == 8312 )
      {
        sub_2CD2F00(a1, a2, (__int64)"__nvvm_coshq", 0xCu, 1);
      }
    }
    else if ( v3 > 0x1FC0 )
    {
      switch ( v3 )
      {
        case 0x1FC1u:
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_acosq", 0xCu, 1);
          break;
        case 0x1FC2u:
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_acoshq", 0xDu, 1);
          break;
        case 0x1FC4u:
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_addq", 0xBu, 2);
          break;
        case 0x1FE8u:
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_asinq", 0xCu, 1);
          break;
        case 0x1FE9u:
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_asinhq", 0xDu, 1);
          break;
        case 0x1FEBu:
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_atanq", 0xCu, 1);
          break;
        case 0x1FECu:
          sub_2CD2F00(a1, a2, (__int64)"__nvvm_atanhq", 0xDu, 1);
          break;
        default:
          return;
      }
    }
  }
}
